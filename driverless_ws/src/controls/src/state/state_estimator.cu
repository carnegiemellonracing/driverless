#include <cuda_utils.cuh>
#include <cuda_globals/cuda_globals.cuh>
#include <glm/glm.hpp>
#include <cuda_constants.cuh>
#include <cmath>


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

            std::cout << "recalculating curvilinear state..." << std::endl;
            recalculate_curv_state();
            std::cout << "done. State: \n";
            for (uint32_t i = 0; i < state_dims; i++) {
                std::cout << m_curv_state[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "-------------------\n" << std::endl;
        }

        void StateEstimator_Impl::on_state(const StateMsg& state_msg) {
            std::cout << "------- ON STATE -----" << std::endl;

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
            
            std::cout << "-------------------\n" << std::endl;
        }

        void StateEstimator_Impl::sync_to_device() {
            std::cout << "sending spline to device texture..." << std::endl;
            send_frames_to_texture();
            std::cout << "done.\n" << std::endl;

            std::cout << "syncing curvilinear state to device..." << std::endl;
            sync_curv_state();
            std::cout << "done.\n" << std::endl;

            // if we allow this to be async, host buffer may be edited by
            // state callbacks during transfer (no bueno)
            cudaDeviceSynchronize();
        }

        std::vector<glm::fvec2> StateEstimator_Impl::get_spline_frames() const {
            std::vector<glm::fvec2> res (m_spline_frames.size());
            for (size_t i = 0; i < m_spline_frames.size(); i++) {
                res[i] = {m_spline_frames[i].x, m_spline_frames[i].y};
            }
            return res;
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

            const float world_yaw = m_world_state[state_yaw_idx];
            const fvec2 world_pos {m_world_state[state_x_idx], m_world_state[state_y_idx]};

            auto distance_to_frame = [world_pos] (const SplineFrame& frame) {
                const fvec2 p {frame.x, frame.y};
                return distance(world_pos, p);
            };


            assert(m_spline_frames.size() > 0);

            float min_dist = distance_to_frame(m_spline_frames[0]);
            size_t min_dist_index = 0;
            for (size_t i = 1; i < m_spline_frames.size(); i++) {
                const SplineFrame& frame = m_spline_frames[i];
                const float dist = distance_to_frame(frame);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_dist_index = i;
                }
            }

            const SplineFrame frame_a = m_spline_frames[min_dist_index];
            const fvec2 a {frame_a.x, frame_a.y};
            const fvec2 a_tangent {cos(frame_a.tangent_angle), sin(frame_a.tangent_angle)};
            const fvec2 a_normal {-a_tangent.y, a_tangent.x};
            const float progress_to_a = min_dist_index * spline_frame_separation;

            const float progress_from_a = dot(a_tangent, world_pos - a);

            const float progress_a = progress_to_a + progress_from_a;
            const float offset_a = dot(a_normal, world_pos - a);
            const float curv_yaw_a = world_yaw - frame_a.tangent_angle;

            float progress;
            float offset;
            float curv_yaw;

            if (progress_from_a != 0
                && !(progress_from_a < 0 && min_dist_index == 0)
                && !(progress_from_a > 0 && min_dist_index == m_spline_frames.size() - 1)) {

                SplineFrame frame_b;
                float progress_to_b;

                if (progress_from_a < 0) {
                    frame_b = m_spline_frames[min_dist_index - 1];
                    progress_to_b = (min_dist_index - 1) * spline_frame_separation;
                } else {  // progress_from_a > 0
                    frame_b = m_spline_frames[min_dist_index + 1];
                    progress_to_b = (min_dist_index + 1) * spline_frame_separation;
                }

                const fvec2 b {frame_b.x, frame_b.y};
                const fvec2 a_to_b = b - a;
                const float a_to_b_dist = length(a_to_b);
                const float t = glm::clamp(dot(a_to_b, world_pos - a) / (a_to_b_dist * a_to_b_dist), 0.0f, 1.0f);

                const fvec2 tangent_b {cos(frame_b.tangent_angle), sin(frame_b.tangent_angle)};
                const fvec2 normal_b {-tangent_b.y, tangent_b.x};

                const float progress_b = progress_to_b + dot(tangent_b, world_pos - b);
                const float offset_b = dot(normal_b, world_pos - b);
                const float curv_yaw_b = world_yaw - frame_b.tangent_angle;

                progress = (1 - t) * progress_a + t * progress_b;
                offset = (1 - t) * offset_a + t * offset_b;
                curv_yaw = (1 - t) * curv_yaw_a + t * curv_yaw_b;
            } else {
                progress = progress_a;
                offset = offset_a;
                curv_yaw = curv_yaw_a;
            }

            m_curv_state[state_x_idx] = progress;
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
