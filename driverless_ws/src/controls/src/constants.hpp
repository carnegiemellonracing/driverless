#pragma once

#include <rclcpp/rclcpp.hpp>


namespace controls {

    /* ROS moments */

    constexpr const char *controller_node_name = "controller";
    constexpr const char *control_action_topic_name = "control_action";
    const rclcpp::QoS control_action_qos (rclcpp::KeepLast(10));


    // MPPI stuff

    /** Controller target frequency, in Hz */
    constexpr double controller_freq = 50.;

    /** Controller target period, in sec */
    constexpr auto controller_period = std::chrono::operator""ms(1000. / controller_freq);
    constexpr uint32_t num_samples = 1024;
    constexpr uint32_t num_timesteps = 128;
    constexpr uint8_t action_dims = 3;
    constexpr uint8_t state_dims = 10;
}