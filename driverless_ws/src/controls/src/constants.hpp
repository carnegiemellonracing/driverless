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

    constexpr uint8_t state_x_idx = 0;
    constexpr uint8_t state_y_idx = 1;
    constexpr uint8_t state_yaw_idx = 2;
    constexpr uint8_t state_car_xdot_idx = 3;
    constexpr uint8_t state_car_ydot_idx = 4;
    constexpr uint8_t state_yawdot_idx = 5;
    constexpr uint8_t state_fz_idx = 6;
    constexpr uint8_t state_mx_idx = 7;
    constexpr uint8_t state_whl_speed_f_idx = 8;
    constexpr uint8_t state_whl_speed_r_idx = 9;

    constexpr uint8_t action_swangle_idx = 0;
    constexpr uint8_t action_torque_f_idx = 1;
    constexpr uint8_t action_torque_r_idx = 2;

    // derived quantities
    constexpr float controller_period_ms = 1000./ controller_freq;
}