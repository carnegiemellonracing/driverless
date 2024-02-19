#pragma once

#include <types.hpp>


namespace controls {
    namespace mppi {
        class MppiController {
        public:
            static std::shared_ptr<MppiController> create();

            virtual Action generate_action() =0;

#ifdef PUBLISH_STATES
            virtual std::vector<float> last_state_trajectories() =0;
#endif

            virtual ~MppiController() = default;
        };

    }
}
