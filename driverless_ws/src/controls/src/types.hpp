#pragma once

#include <interfaces/msg/spline_frames.hpp>
#include <array>
#include <interfaces/msg/control_action.hpp>
#include <interfaces/msg/controls_state.hpp>

#include "constants.hpp"


namespace controls {
    using Action = std::array<float, action_dims>;
    using State = std::array<float, state_dims>;

    using SplineMsg = interfaces::msg::SplineFrames;
    using StateMsg = interfaces::msg::ControlsState;
    using ActionMsg = interfaces::msg::ControlAction;

    class Controller {
    public:
        virtual Action generate_action() =0;

        virtual ~Controller() =0;
    };
}
