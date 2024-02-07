#include <cuda_utils.cuh>
#include <thrust/device_free.h>
#include <cuda_globals/cuda_globals.cuh>

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

        StateEstimator_Impl::StateEstimator_Impl() {
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


            // initialize curr state double buffering
            {
                std::lock_guard<std::mutex> guard {cuda_globals::state_swapping_mutex};

                CUDA_CALL(cudaGetSymbolAddress(
                    reinterpret_cast<void**>(&cuda_globals::curr_state_read),
                    &cuda_globals::curr_state_buf1
                ));

                CUDA_CALL(cudaGetSymbolAddress(
                    reinterpret_cast<void**>(&cuda_globals::curr_state_write),
                    &cuda_globals::curr_state_buf2
                ));
            }
        }

        StateEstimator_Impl::~StateEstimator_Impl() {
            assert(cuda_globals::spline_texture_created);

            cudaDestroyTextureObject(cuda_globals::spline_texture_object);
            cudaFreeArray(cuda_globals::spline_array);

            cuda_globals::spline_texture_created = false;
        }

        void StateEstimator_Impl::on_spline(const SplineMsg& spline_msg) {
            m_host_spline_frames.clear();
            m_host_spline_frames.reserve(spline_msg.frames.size());

            for (const interfaces::msg::SplineFrame& frame : spline_msg.frames) {
                m_host_spline_frames.push_back(SplineFrame {frame.x, frame.y, 0.0f, 0.0f});
            }

            populate_tangent_angles(m_host_spline_frames);
            populate_curvatures(m_host_spline_frames);

            assert(cuda_globals::spline_texture_created);

            send_frames_to_texture();
            recalculate_state();
            sync_state();
        }

        void StateEstimator_Impl::on_slam(const SlamMsg& slam_msg) {
            // TODO: update state based on slam

            sync_state();
        }

        void StateEstimator_Impl::send_frames_to_texture() {
            assert(cuda_globals::spline_texture_created);

            size_t elems = m_host_spline_frames.size();
            assert(elems <= max_spline_texture_elems);
            CUDA_CALL(cudaMemcpyToSymbolAsync(&cuda_globals::spline_texture_elems, &elems, sizeof(elems)));

            CUDA_CALL(cudaMemcpy2DToArrayAsync(
                cuda_globals::spline_array, 0, 0,
                m_host_spline_frames.data(),
                elems * sizeof(SplineFrame), elems * sizeof(SplineFrame), 1,
                cudaMemcpyHostToDevice
            ));
        }

        void StateEstimator_Impl::recalculate_state() {
            // TODO: calculate curvilinear state using ternary search
        }

        void StateEstimator_Impl::sync_state() {
            CUDA_CALL(cudaMemcpyToSymbolAsync(
                &cuda_globals::curr_state, &m_host_curv_state, state_dims * sizeof(float)
            ));
        }

    }
}
