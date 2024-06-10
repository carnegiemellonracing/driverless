#pragma once

#include <interfaces/msg/spline_frames.hpp>
#include <array>
#include <interfaces/msg/control_action.hpp>
#include <interfaces/msg/controls_state.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/quaternion_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <interfaces/msg/controller_info.hpp>

#include "constants.hpp"

namespace controls {
    /// Control action: currently steering wheel angle and forward throttle
    using Action = std::array<float, action_dims>;
    /// Vehicle state: currently inertial x, y, yaw, speed //TODO: confirm
    using State = std::array<float, state_dims>;

    /// ROS Messages
    using SplineMsg = interfaces::msg::SplineFrames;
    using TwistMsg = geometry_msgs::msg::TwistStamped;
    using ActionMsg = interfaces::msg::ControlAction;
    using QuatMsg = geometry_msgs::msg::QuaternionStamped;
    using PoseMsg = geometry_msgs::msg::PoseStamped;
    using StateMsg = interfaces::msg::ControlsState;
    using InfoMsg = interfaces::msg::ControllerInfo;

    /// Logging function type.
    using LoggerFunc = std::function<void(const char*)>;
    /// Instance of LoggerFunc that doesn't log anything.
    constexpr void no_log(const char*) {};

    /// TODO: DELETE
//    class Controller {
//    public:
//        virtual Action generate_action() =0;
//
//        virtual ~Controller() =0;
//    };
}
