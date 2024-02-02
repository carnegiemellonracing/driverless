#pragma once

#include <interfaces/msg/spline_list.hpp>
#include <array>

#include "constants.hpp"


namespace controls {
    using Action = std::array<double, action_dims>;
    using State = std::array<double, state_dims>;

    using SplineMsg = interfaces::msg::SplineList;
    using SlamMsg = struct {};

    class Controller {
    public:
        virtual Action generate_action(const State &current_state) =0;

        virtual ~Controller() =0;
    };
}
