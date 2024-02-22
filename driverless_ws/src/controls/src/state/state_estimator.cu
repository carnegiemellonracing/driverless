#include <cuda_utils.cuh>
#include <cuda_globals/cuda_globals.cuh>
#include <glm/glm.hpp>
#include <cuda_constants.cuh>


#include "state_estimator.cuh"
#include "state_estimator.hpp"



namespace controls {
    namespace state {

        std::shared_ptr<StateEstimator> StateEstimator::create() {
            return std::make_shared<StateEstimator_Impl>();
        }


        // StateEstimator_Impl helpers

        void populate_tangent_angles(std::vector<SplineFrame>& frames) {
            assert(frames.size() > 1);

            auto angle_from_to = [] (const SplineFrame& a, const SplineFrame& b) {
                const float dx = b.x - a.x;
                const float dy = b.y - a.y;
                return std::atan2(dy, dx);
            };

            frames[0].tangent_angle = angle_from_to(frames[0], frames[1]);
            for (size_t i = 1; i < frames.size() - 1; i++) {
                frames[i].tangent_angle = angle_from_to(frames[i - 1], frames[i + 1]);
            }
            frames.back().tangent_angle = angle_from_to(*(frames.end() - 2), frames.back());
        }

        void populate_curvatures(std::vector<SplineFrame>& frames) {
            assert(frames.size() > 1);

            auto curvature_from_to = [] (const SplineFrame& a, const SplineFrame& b, uint8_t samples_apart) {
                const float dphi = b.tangent_angle - a.tangent_angle;
                return dphi / (samples_apart * spline_frame_separation);
            };

            frames[0].curvature = curvature_from_to(frames[0], frames[1], 1);
            for (size_t i = 1; i < frames.size() - 1; i++) {
                frames[i].curvature = curvature_from_to(frames[i - 1], frames[i + 1], 2);
            }
            frames.back().curvature = curvature_from_to(*(frames.end() - 2), frames.back(), 1);
        }

        // methods

        StateEstimator_Impl::StateEstimator_Impl()
            : m_curv_state {}, m_world_state {} {

            assert(!cuda_globals::spline_texture_created);

            cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(
                32, 32, 32, 32, cudaChannelFormatKindFloat
            );

            CUDA_CALL(cudaMalloc(&cuda_globals::spline_texture_buf, sizeof(float4) * max_spline_texture_elems));

            cudaResourceDesc resource_desc {};
            resource_desc.resType = cudaResourceTypeLinear;
            resource_desc.res.linear.desc = channel_desc;
            resource_desc.res.linear.devPtr = cuda_globals::spline_texture_buf;
            resource_desc.res.linear.sizeInBytes = max_spline_texture_elems * sizeof(float4);

            cudaTextureDesc texture_desc {};
            texture_desc.addressMode[0] = cudaAddressModeClamp;
            texture_desc.filterMode = cudaFilterModePoint;
            texture_desc.readMode = cudaReadModeElementType;
            texture_desc.normalizedCoords = false;

            CUDA_CALL(cudaCreateTextureObject(
                &cuda_globals::spline_texture_object, &resource_desc, &texture_desc, nullptr
            ));

            CUDA_CALL(cudaMemcpyToSymbol(
                cuda_globals::d_spline_texture_object, &cuda_globals::spline_texture_object,
                sizeof(cuda_globals::spline_texture_object)
            ));

            cuda_globals::spline_texture_created = true;
        }

        StateEstimator_Impl::~StateEstimator_Impl() {
            assert(cuda_globals::spline_texture_created);

            cudaDestroyTextureObject(cuda_globals::spline_texture_object);
            cudaFree(cuda_globals::spline_texture_buf);

            cuda_globals::spline_texture_created = false;
        }

        void StateEstimator_Impl::on_spline(const SplineMsg& spline_msg) {
            std::cout << "------- ON SPLINE -----" << std::endl;

            m_spline_frames.clear();
            m_spline_frames.reserve(spline_msg.frames.size());

            for (const auto& frame : spline_msg.frames) {
                m_spline_frames.push_back(SplineFrame {frame.x, frame.y, 0.0f, 0.0f});
            }

            std::cout << "populating tangent angles..." << std::endl;
            populate_tangent_angles(m_spline_frames);
            std::cout << "done.\n" << std::endl;

            std::cout << "populating curvatures..." << std::endl;
            populate_curvatures(m_spline_frames);
            std::cout << "done.\n" << std::endl;

            assert(cuda_globals::spline_texture_created);

            std::cout << "sending spline to device texture..." << std::endl;
            send_frames_to_texture();
            std::cout << "done.\n" << std::endl;

            std::cout << "recalculating curvilinear state..." << std::endl;
            recalculate_curv_state();
            std::cout << "done. State: \n";
            for (uint32_t i = 0; i < state_dims; i++) {
                std::cout << m_curv_state[i] << " ";
            }
            std::cout << std::endl;

            std::cout << "syncing curvilinear state to device..." << std::endl;
            sync_curv_state();
            std::cout << "done.\n" << std::endl;

            std::cout << "-------------------\n" << std::endl;
        }

        void StateEstimator_Impl::on_state(const StateMsg& state_msg) {
            m_world_state[state_x_idx] = state_msg.x;
            m_world_state[state_y_idx] = state_msg.y;
            m_world_state[state_yaw_idx] = state_msg.yaw;
            m_world_state[state_car_xdot_idx] = state_msg.xcar_dot;
            m_world_state[state_car_ydot_idx] = state_msg.ycar_dot;
            m_world_state[state_yawdot_idx] = state_msg.yaw_dot;
            m_world_state[state_my_idx] = state_msg.moment_y;
            m_world_state[state_fz_idx] = state_msg.downforce;
            m_world_state[state_whl_speed_f_idx] = state_msg.whl_speed_f;
            m_world_state[state_whl_speed_r_idx] = state_msg.whl_speed_r;

            recalculate_curv_state();
            sync_curv_state();
        }

        std::vector<SplineFrame> StateEstimator_Impl::get_spline_frames() const {
            return m_spline_frames;
        }

        void StateEstimator_Impl::send_frames_to_texture() {
            assert(cuda_globals::spline_texture_created);

            size_t elems = m_spline_frames.size();
            assert(elems <= max_spline_texture_elems);
            CUDA_CALL(cudaMemcpyToSymbolAsync(cuda_globals::spline_texture_elems, &elems, sizeof(elems)));

            CUDA_CALL(cudaMemcpy(
                cuda_globals::spline_texture_buf, m_spline_frames.data(),
                sizeof(SplineFrame) * elems,
                cudaMemcpyHostToDevice
            ));
        }

        void StateEstimator_Impl::recalculate_curv_state() {
            using namespace glm;

            const fvec2 world_pos {m_world_state[state_x_idx], m_world_state[state_y_idx]};

            float min_dist_progress;
            float min_dist_offset;
            float min_dist_curv_yaw;
            float min_dist;
            bool found = false;
            for (size_t i = 0; i < m_spline_frames.size() - 1; i++) {
                SplineFrame frame1 = m_spline_frames[i];
                SplineFrame frame2 = m_spline_frames[i + 1];

                const fvec2 frame1pos {frame1.x, frame1.y};
                const fvec2 frame2pos {frame2.x, frame2.y};
                const fvec2 segment_disp = frame2pos - frame1pos;
                const fvec2 tangent = normalize(segment_disp);
                const fvec2 car_disp = world_pos - frame1pos;
                const float closest_progress_on_line = dot(tangent, car_disp);
                if (closest_progress_on_line < 0 || closest_progress_on_line >= segment_disp.length()) {
                    continue;
                }

                const fvec2 closest_point = frame1pos + closest_progress_on_line * tangent;
                const float car_dist = distance(closest_point, world_pos);

                if (!found || car_dist < min_dist) {
                    found = true;

                    min_dist = car_dist;
                    min_dist_progress = i * spline_frame_separation + closest_progress_on_line;
                    min_dist_offset = car_dist;
                    min_dist_curv_yaw = m_world_state[state_yaw_idx] - atan2(tangent.y, tangent.x);
                }
            }

            assert(found);

            m_curv_state[state_x_idx] = min_dist_progress;
            m_curv_state[state_y_idx] = min_dist_offset;
            m_curv_state[state_yaw_idx] = min_dist_curv_yaw;
            std::copy(&m_world_state[3], m_world_state.end(), &m_curv_state[3]);
        }

        void StateEstimator_Impl::sync_curv_state() {
            CUDA_CALL(cudaMemcpyToSymbolAsync(
                cuda_globals::curr_state, &m_curv_state, state_dims * sizeof(float)
            ));
        }

    }
}
