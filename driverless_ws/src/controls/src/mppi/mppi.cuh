#pragma once

#include <thrust/device_ptr.h>

#include "mppi.hpp"


namespace controls {
    namespace mppi {

        class MppiController : public Controller {
        public:
            MppiController();

            Action generate_action(const State& current_state) override;

            ~MppiController() override;

        private:
            thrust::device_ptr<float> m_action_trajectories;
            thrust::device_ptr<float> m_cost_to_gos;
            thrust::device_ptr<float> m_last_action_trajectory;
        };

    }
}