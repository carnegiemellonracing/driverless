#pragma once

#include <rclcpp/rclcpp.hpp>

#include "types.hpp"


namespace controls {
    constexpr const char *controller_node_name = "controller";
    constexpr const char *control_action_topic_name = "control_action";

    constexpr Device default_device = Device::Cuda;

    /** Controller target frequency, in Hz */
    constexpr double controller_freq = 50.;

    /** Controller target period, in sec */
    constexpr auto controller_period = std::chrono::operator""ms(1000. / controller_freq);

    constexpr rclcpp::QoS control_action_qos (rclcpp::KeepLast(10));
}