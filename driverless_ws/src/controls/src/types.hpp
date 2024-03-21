#pragma once

#include <interfaces/msg/spline_frames.hpp>
#include <array>
#include <interfaces/msg/control_action.hpp>
#include <interfaces/msg/controls_state.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include "constants.hpp"

namespace controls {
    using Action = std::array<float, action_dims>;
    using State = std::array<float, state_dims>;

    using SplineMsg = interfaces::msg::SplineFrames;
    using TwistMsg = geometry_msgs::msg::TwistStamped;
    using ActionMsg = interfaces::msg::ControlAction;
    using QuatMsg = geometry_msgs::msg::QuaternionStamped;
    using PoseMsg = geometry_msgs::msg::PoseStamped;
    using StateMsg = interfaces::msg::ControlsState;

    class Controller {
    public:
        virtual Action generate_action() =0;

        virtual ~Controller() =0;
    };
}
