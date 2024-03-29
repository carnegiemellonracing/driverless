#pragma once

#include <rclcpp/rclcpp.hpp>

namespace controls {
    /* ROS moments */

    constexpr const char *controller_node_name = "controller";
    constexpr const char *control_action_topic_name = "control_action";
    constexpr const char *spline_topic_name = "spline";
    constexpr const char *state_topic_name = "state";
    constexpr const char *world_twist_topic_name = "filter/twist";
    constexpr const char *world_quat_topic_name = "filter/quaternion";
    constexpr const char *world_pose_topic_name = "filter/pose";
    constexpr const char *controller_info_topic_name = "controller_info";

    const rclcpp::QoS control_action_qos (rclcpp::KeepLast(10));
    const rclcpp::QoS spline_qos (rclcpp::KeepLast(1));
    const rclcpp::QoS state_qos (rclcpp::KeepLast(1));
    const rclcpp::QoS world_twist_qos (rclcpp::KeepLast(1));
    const rclcpp::QoS world_quat_qos (rclcpp::KeepLast(1));
    const rclcpp::QoS world_pose_qos (rclcpp::KeepLast(1));
    const rclcpp::QoS controller_info_qos (rclcpp::KeepLast(10));


    // MPPI stuff

    /** Controller target frequency, in Hz */
    constexpr double controller_freq = 50.;
    constexpr float controller_period = 1. / controller_freq;

    constexpr double controller_publish_freq = 50.;
    constexpr float controller_publish_period = 1. / controller_publish_freq;

    /** Controller target period, in sec */
    constexpr uint32_t num_samples = 1024 * 8;
    constexpr uint32_t num_timesteps = 96;
    constexpr uint8_t action_dims = 2;
    constexpr uint8_t state_dims = 4;
    constexpr float temperature = 1.0f;
    constexpr unsigned long long seed = 0;
    constexpr uint32_t num_action_trajectories = action_dims * num_timesteps * num_samples;

    constexpr float init_action_trajectory[num_timesteps * action_dims] = {};


    // Cost params

    constexpr float offset_1m_cost = 2.0f;
    constexpr float target_speed = 3.0f;
    constexpr float no_speed_cost = 5.0f;
    constexpr float overspeed_1m_cost = 1.0f;
    constexpr float torque_100N_per_sec_cost = 0.0f;


    // State Estimation

    constexpr float spline_frame_separation = 0.5f;  // meters
    constexpr uint32_t curv_frame_lookup_tex_width = 512;
    constexpr float curv_frame_lookup_padding = 0; // meters
    constexpr float track_width = 5.0f;
    constexpr float car_padding = spline_frame_separation;
    constexpr bool should_estimate_whl_speeds = false;


    // Car params

    constexpr float cg_to_front = 0.775;
    constexpr float cg_to_rear = 0.775;
    constexpr float whl_radius = 0.2286;
    constexpr float gear_ratio = 15.0f;
    constexpr float car_mass = 310.0f;
    constexpr float rolling_drag = 200.0f; // N
    constexpr float long_tractive_capability = 5.0f; // m/s^2 
    constexpr float lat_tractive_capability = 5.0f; // m/s^2
    constexpr float understeer_slope = 0.025f;
    constexpr float brake_enable_speed = 2.0f;
    constexpr float saturating_motor_torque = long_tractive_capability * car_mass * whl_radius / gear_ratio;

    enum class TorqueMode
    {
        AWD,
        FWD,
        RWD
    };
    constexpr TorqueMode torque_mode = TorqueMode::FWD;


    // Indices

    constexpr uint8_t state_x_idx = 0;
    constexpr uint8_t state_y_idx = 1;
    constexpr uint8_t state_yaw_idx = 2;
    constexpr uint8_t state_speed_idx = 3;

    constexpr uint8_t action_swangle_idx = 0;
    constexpr uint8_t action_torque_idx = 1;


    // misc

    constexpr const char* clear_term_sequence = "\033c";
}