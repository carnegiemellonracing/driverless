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
            void on_state(const StateMsg& state_msg) override;

            void sync_to_device() override;

            std::vector<glm::fvec2> get_spline_frames() override;

            ~StateEstimator_Impl() override;

        private:
            void send_frames_to_texture();
            void recalculate_curv_state();
            void sync_curv_state();

            std::vector<SplineFrame> m_spline_frames;

            State m_curv_state;
            State m_world_state;
        };

    }
}
