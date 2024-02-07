#pragma once

#include <cuda_types.cuh>
#include <thrust/device_vector.h>

#include "state_estimator.hpp"


namespace controls {
    namespace state {

        class StateEstimator_Impl : public StateEstimator {
        public:
            StateEstimator_Impl();

            void on_spline(const SplineMsg& spline_msg) override;
            void on_slam(const SlamMsg& slam_msg) override;

            ~StateEstimator_Impl() override;

        private:
            void send_frames_to_texture();
            void recalculate_state();
            void sync_state();

            std::vector<SplineFrame> m_host_spline_frames;
            State m_host_curv_state;
        };

    }
}
