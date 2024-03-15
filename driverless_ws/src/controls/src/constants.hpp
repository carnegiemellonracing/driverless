#pragma once

#include <rclcpp/rclcpp.hpp>

namespace controls {

    /* ROS moments */

    constexpr const char *controller_node_name = "controller";
    constexpr const char *control_action_topic_name = "control_action";
    constexpr const char *spline_topic_name = "spline";
    constexpr const char *state_topic_name = "state";
    constexpr const char *state_trajectories_topic_name = "state_trajectories";
    const rclcpp::QoS control_action_qos (rclcpp::KeepLast(10));
    const rclcpp::QoS spline_qos (rclcpp::KeepLast(1));
    const rclcpp::QoS state_qos (rclcpp::KeepLast(1));
    const rclcpp::QoS state_trajectories_qos (rclcpp::KeepLast(10));

    // MPPI stuff

    /** Controller target frequency, in Hz */
    constexpr double controller_freq = 50.;
    constexpr float controller_period = 1. / controller_freq;

    constexpr double controller_publish_freq = 10.;
    constexpr float controller_publish_period = 1. / controller_publish_freq;


    /** Controller target period, in sec */
    constexpr uint32_t num_samples = 1024 * 8;
    constexpr uint32_t num_timesteps = 96;
    constexpr uint8_t action_dims = 2;
    constexpr uint8_t state_dims = 10;
    constexpr float temperature = 0.5f;
    constexpr unsigned long long seed = 0;
    constexpr uint32_t num_action_trajectories = action_dims * num_timesteps * num_samples;

    constexpr float init_action_trajectory[num_timesteps * action_dims] = {};

    // Cost params
    constexpr float offset_1m_cost = 1.0f;
    constexpr float target_speed = 7.5f;
    constexpr float speed_weight = 0.05f;


    // State Estimation

    constexpr float spline_frame_separation = 0.5f;  // meters
    constexpr uint32_t curv_frame_lookup_tex_width = 512;
    constexpr float curv_frame_lookup_padding = 0; // meters
    constexpr float track_width = 5.0f;
    constexpr float car_padding = 3.0f;

    constexpr uint8_t state_x_idx = 0;
    constexpr uint8_t state_y_idx = 1;
    constexpr uint8_t state_yaw_idx = 2;
    constexpr uint8_t state_car_xdot_idx = 3;
    constexpr uint8_t state_car_ydot_idx = 4;
    constexpr uint8_t state_yawdot_idx = 5;
    constexpr uint8_t state_my_idx = 6;
    constexpr uint8_t state_fz_idx = 7;
    constexpr uint8_t state_whl_speed_f_idx = 8;
    constexpr uint8_t state_whl_speed_r_idx = 9;

    constexpr uint8_t action_swangle_idx = 0;
    constexpr uint8_t action_torque_idx = 1;

}