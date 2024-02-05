#pragma once

#include <types.hpp>


namespace controls {
    namespace mppi {

        class MppiController {
        public:
            virtual Action generate_action() =0;

            virtual ~MppiController() =0;

            static std::unique_ptr<MppiController> create();
        };

    }
}
