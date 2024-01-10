#pragma once

#include <types.hpp>

#include "mppi.hpp"


namespace controls {
    namespace mppi {
        class CpuMppiController : public MppiController {
        public:
            Action generate_action(const State& current_state) override;
        };
    }
}