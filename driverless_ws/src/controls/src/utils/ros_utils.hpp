#include <rclcpp/rclcpp.hpp>
#include "../types.hpp"

namespace controls {

    inline float twist_msg_to_speed(const TwistMsg& twist_msg) {
        return std::sqrt(
        twist_msg.twist.linear.x * twist_msg.twist.linear.x
        + twist_msg.twist.linear.y * twist_msg.twist.linear.y);
    }
}