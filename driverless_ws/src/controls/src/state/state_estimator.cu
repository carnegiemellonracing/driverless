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

        std::unique_ptr<StateEstimator> StateEstimator::create() {
            return std::make_unique<StateEstimator_Impl>();
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
                frames[i].tangent_angle = curvature_from_to(frames[i - 1], frames[i + 1], 2);
            }
            frames.back().tangent_angle = curvature_from_to(*(frames.end() - 2), frames.back(), 1);
        }

        // methods

        StateEstimator_Impl::StateEstimator_Impl()
            : m_curv_state {}, m_world_state {} {

            assert(!cuda_globals::spline_texture_created);

            cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(
                32, 32, 32, 32, cudaChannelFormatKindFloat
            );

            CUDA_CALL(cudaMallocArray(&cuda_globals::spline_array, &channel_desc, max_spline_texture_elems, 1));

            cudaResourceDesc resource_desc {};
            resource_desc.resType = cudaResourceTypeArray;
            resource_desc.res.array.array = cuda_globals::spline_array;

            cudaTextureDesc texture_desc {};
            texture_desc.addressMode[0] = cudaAddressModeClamp;
            texture_desc.filterMode = cudaFilterModeLinear;
            texture_desc.readMode = cudaReadModeElementType;
            texture_desc.normalizedCoords = false;

            CUDA_CALL(cudaCreateTextureObject(
                &cuda_globals::spline_texture_object, &resource_desc, &texture_desc, nullptr
            ));

            cuda_globals::spline_texture_created = true;
        }

        StateEstimator_Impl::~StateEstimator_Impl() {
            assert(cuda_globals::spline_texture_created);

            cudaDestroyTextureObject(cuda_globals::spline_texture_object);
            cudaFreeArray(cuda_globals::spline_array);

            cuda_globals::spline_texture_created = false;
        }

        void StateEstimator_Impl::on_spline(const SplineMsg& spline_msg) {
            m_spline_frames.clear();
            m_spline_frames.reserve(spline_msg.frames.size());

            for (const interfaces::msg::SplineFrame& frame : spline_msg.frames) {
                m_spline_frames.push_back(SplineFrame {frame.x, frame.y, 0.0f, 0.0f});
            }

            populate_tangent_angles(m_spline_frames);
            populate_curvatures(m_spline_frames);

            assert(cuda_globals::spline_texture_created);

            send_frames_to_texture();
            recalculate_curv_state();
            sync_curv_state();
        }

        void StateEstimator_Impl::on_slam(const SlamMsg& slam_msg) {
            m_world_state[state_x_idx] = slam_msg.x;
            m_world_state[state_y_idx] = slam_msg.y;
            m_world_state[state_yaw_idx] = slam_msg.theta;

            recalculate_curv_state();
            sync_curv_state();
        }

        void StateEstimator_Impl::send_frames_to_texture() {
            assert(cuda_globals::spline_texture_created);

            size_t elems = m_spline_frames.size();
            assert(elems <= max_spline_texture_elems);
            CUDA_CALL(cudaMemcpyToSymbolAsync(&cuda_globals::spline_texture_elems, &elems, sizeof(elems)));

            CUDA_CALL(cudaMemcpy2DToArrayAsync(
                cuda_globals::spline_array, 0, 0,
                m_spline_frames.data(),
                elems * sizeof(SplineFrame), elems * sizeof(SplineFrame), 1,
                cudaMemcpyHostToDevice
            ));
        }

        void StateEstimator_Impl::recalculate_curv_state() {
            using namespace glm;

            auto distance_to_segment = [this] (fvec2 pos, size_t segment) {
                const SplineFrame frame = m_spline_frames[segment];
                return length(pos - fvec2(frame.x, frame.y));
            };

            size_t left = 0;
            size_t right = m_spline_frames.size();

            const fvec2 world_pos {m_world_state[state_x_idx], m_world_state[state_y_idx]};

            while (left != right) {
                assert(left < right);

                const size_t left_check = left + (right - left) / 3;
                const size_t right_check = right - (right - left) / 3;

                if (distance_to_segment(world_pos, left_check) < distance_to_segment(world_pos, right_check)) {
                    right = right_check;
                } else {
                    left = left_check;
                }
            }

            const size_t segment = left;
            const SplineFrame frame = m_spline_frames[segment];
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

            m_curv_state[state_x_idx] = segment * spline_frame_separation + arc_progress;
            m_curv_state[state_y_idx] = offset;
            m_curv_state[state_yaw_idx] = curv_yaw;
            std::copy(&m_world_state[3], m_world_state.end(), &m_curv_state[3]);
        }

        void StateEstimator_Impl::sync_curv_state() {
            CUDA_CALL(cudaMemcpyToSymbolAsync(
                &cuda_globals::curr_state, &m_curv_state, state_dims * sizeof(float)
            ));
        }

    }
}
