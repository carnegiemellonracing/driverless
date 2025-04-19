#pragma once

#include <array>

#include "constants.hpp"
#include "ros_types_and_constants.hpp"

namespace controls {
    /// Control action: currently steering wheel angle and forward throttle
    using Action = std::array<float, action_dims>;
    /// Vehicle state: currently inertial x, y, yaw, speed, requested swangle
    using State = std::array<float, state_dims>;



    /// Logging function type.
    using LoggerFunc = std::function<void(const char*)>;
    /// Instance of LoggerFunc that doesn't log anything.
    constexpr void no_log(const char*) {};

    using XYPosition = std::pair<float, float>;
    using PositionAndYaw = std::pair<XYPosition, float>;
}
