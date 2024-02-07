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
    constexpr uint32_t num_samples = 1024;
    constexpr uint32_t num_timesteps = 128;
    constexpr uint8_t action_dims = 3;
    constexpr uint8_t state_dims = 10;
    constexpr uint32_t num_spline_frames = 128;
    constexpr float temperature = 1.0f;
    constexpr unsigned long long seed = 0;
    constexpr uint32_t num_action_trajectories = action_dims * num_timesteps * num_samples;


    // State Estimation

    constexpr float spline_frame_separation = 0.5f;  // meters
    constexpr std::vector<float> finite_difference_smoothing_filter = {0.25, 0.5, 0.25}; // odd

    static_assert(spline_finite_difference_smoothing_samples % 2 == 1);

    // derived quantities
    constexpr float controller_period_ms = 1000./ controller_freq;
}