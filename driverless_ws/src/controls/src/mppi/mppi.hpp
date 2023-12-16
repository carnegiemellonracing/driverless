#pragma once

#include <common/types.hpp>

namespace controls {
    namespace mppi {
        class mppi_controller : public controller {
        public:
            mppi_controller () = default;  // not strictly necessary but emphasizes
                                           // this is the intended way to construct

            action generate_action(const state &current_state) override;
        };
    }
}