#include <cuda_utils.cuh>
#include <interfaces/msg/spline_frame_list.hpp>
#include <interfaces/msg/spline_frame.hpp>
#include <cuda_globals/cuda_globals.cuh>
#include <glm/glm.hpp>

#include "state_estimator.cuh"
#include "state_estimator.hpp"

#include <cuda_constants.cuh>


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
            m_world_state[state_mx_idx] = state_msg.moment_y;
            m_world_state[state_fz_idx] = state_msg.downforce;
            m_world_state[state_whl_speed_f_idx] = state_msg.whl_speed_f;
            m_world_state[state_whl_speed_r_idx] = state_msg.whl_speed_r;

            recalculate_curv_state();
            sync_curv_state();
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

            auto distance_to_segment = [this] (fvec2 pos, size_t segment) {
                const SplineFrame frame = m_spline_frames[segment];
                return length(pos - fvec2(frame.x, frame.y));
            };

            const fvec2 world_pos {m_world_state[state_x_idx], m_world_state[state_y_idx]};

            size_t min_dist_segment = 0;
            float min_dist = distance_to_segment(world_pos, min_dist_segment);
            for (size_t i = 0; i < m_spline_frames.size(); i++) {
                const float this_dist = distance_to_segment(world_pos, i);
                if (this_dist < min_dist) {
                    min_dist = this_dist;
                    min_dist_segment = i;
                }
            }

            const SplineFrame frame = m_spline_frames[min_dist_segment];
            const fvec2 segment_start {frame.x, frame.y};

            float arc_progress;
            float offset;
            float curv_yaw;

            if (frame.curvature == 0) {
                const fvec2 normal {sin(frame.tangent_angle), -cos(frame.tangent_angle)};
                offset = dot(world_pos - segment_start, normal);
                arc_progress = length(world_pos - normal * offset - segment_start);
                curv_yaw = m_world_state[state_yaw_idx] - frame.tangent_angle;

            } else {
                const float angle_to_center = frame.tangent_angle + radians(90.);
                const fvec2 center_to_start = -fvec2 {cos(angle_to_center), sin(angle_to_center)} / frame.curvature;

                const fvec2 center = segment_start - center_to_start;
                const fvec2 center_to_pos = world_pos - center;

                const float vecs_crossed = center_to_start.x * center_to_pos.y - center_to_pos.x * center_to_start.y;
                const float angle_along_arc = asin(frame.curvature * vecs_crossed / length(center_to_pos));

                arc_progress = angle_along_arc / abs(frame.curvature);
                offset = length(center_to_start) - length(center_to_pos);
                curv_yaw = m_world_state[state_yaw_idx] - (frame.tangent_angle + arc_progress * frame.curvature);
            }

            assert(arc_progress >= 0);

            m_curv_state[state_x_idx] = min_dist_segment * spline_frame_separation + arc_progress;
            m_curv_state[state_y_idx] = offset;
            m_curv_state[state_yaw_idx] = curv_yaw;
            std::copy(&m_world_state[3], m_world_state.end(), &m_curv_state[3]);
        }

        void StateEstimator_Impl::sync_curv_state() {
            CUDA_CALL(cudaMemcpyToSymbolAsync(
                cuda_globals::curr_state, &m_curv_state, state_dims * sizeof(float)
            ));
        }

    }
}
