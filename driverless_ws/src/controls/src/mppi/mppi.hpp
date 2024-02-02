#pragma once

#include <types.hpp>


namespace controls {
    namespace mppi {

        std::unique_ptr<Controller> make_mppi_controller();

    }
}
