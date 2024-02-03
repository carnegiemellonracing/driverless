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

    // derived quantities
    constexpr float controller_period_ms = 1000./ controller_freq;
    constexpr dim3 action_trajectories_dims {num_samples, num_timesteps, action_dims};

    // brownian stuff
    constexpr curandRngType_t rng_type = CURAND_RNG_PSEUDO_MTGP32;
    constexpr unsigned long long seed = 0;
    constexpr uint32_t num_action_trajectories = (uint32_t) action_dims * num_timesteps * num_samples; // TODO: this casting ok?
}