#pragma once

#include <types.hpp>


namespace controls {
    namespace mppi {

        class MppiController {
        public:
            static std::unique_ptr<MppiController> create();

            virtual Action generate_action() =0;

            virtual ~MppiController() = default;
        };

    }
}
