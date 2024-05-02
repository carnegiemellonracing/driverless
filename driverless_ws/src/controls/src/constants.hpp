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

    static const rmw_qos_profile_t best_effort_profile = {
        RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        1,
        RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
        RMW_QOS_POLICY_DURABILITY_VOLATILE,
        RMW_QOS_DEADLINE_DEFAULT,
        RMW_QOS_LIFESPAN_DEFAULT,
        RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
        RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
        false
    };
    const rclcpp::QoS best_effort_qos = rclcpp::QoS(
        rclcpp::QoSInitialization(
          best_effort_profile.history,
          best_effort_profile.depth
        ),
        best_effort_profile);
    
    const rclcpp::QoS control_action_qos = best_effort_qos;
    const rclcpp::QoS spline_qos = best_effort_qos;
    const rclcpp::QoS state_qos (rclcpp::KeepLast(1));
    const rclcpp::QoS world_twist_qos (rclcpp::KeepLast(1));
    const rclcpp::QoS world_quat_qos (rclcpp::KeepLast(1));
    const rclcpp::QoS world_pose_qos (rclcpp::KeepLast(1));
    const rclcpp::QoS controller_info_qos = best_effort_qos;

    constexpr rcl_clock_type_t default_clock_type = RCL_ROS_TIME;


    // MPPI stuff

    /** Controller target frequency, in Hz */
    constexpr double controller_freq = 10.;
    constexpr float controller_period = 1. / controller_freq;

    constexpr double controller_publish_freq = controller_freq;
    constexpr float controller_publish_period = 1. / controller_publish_freq;

    /** Controller target period, in sec */
    constexpr uint32_t num_samples = 1024 * 64;
    constexpr uint32_t num_timesteps = 16;
    constexpr uint8_t action_dims = 2;
    constexpr uint8_t state_dims = 4;
    constexpr float temperature = 1.0f;
    constexpr unsigned long long seed = 0;
    constexpr uint32_t num_action_trajectories = action_dims * num_timesteps * num_samples;
    constexpr float init_action_trajectory[num_timesteps * action_dims] = {};
    constexpr float action_momentum = 0.0f;

    // Cost params

    constexpr float offset_1m_cost = 5.0f;
    constexpr float target_speed = 13.0f;
    constexpr float speed_off_1mps_cost = 1.0f;
    constexpr float out_of_bounds_cost = 100.0f;


    // State Estimation

    constexpr float spline_frame_separation = 0.5f;  // meters
    constexpr uint32_t curv_frame_lookup_tex_width = 512;
    constexpr float curv_frame_lookup_padding = 0; // meters
    constexpr float track_width = 30.0f;
    constexpr float car_padding = std::max(spline_frame_separation, M_SQRT2f32 * track_width);
    constexpr bool reset_pose_on_spline = true;


    // Car params

    constexpr float cg_to_front = 0.775;
    constexpr float cg_to_rear = 0.775;
    constexpr float cg_to_nose = 1.5f;
    constexpr float whl_base = 2.0f;
    constexpr float whl_radius = 0.2286;
    constexpr float gear_ratio = 15.0f;
    constexpr float car_mass = 210.0f;
    constexpr float rolling_drag = 100.0f; // N
    constexpr float long_tractive_capability = 3.5f; // m/s^2
    constexpr float lat_tractive_capability = 5.0f; // m/s^2
    constexpr float understeer_slope = 0.0f;
    constexpr float brake_enable_speed = 1.0f;
    constexpr float saturating_motor_torque = (long_tractive_capability + rolling_drag / car_mass) * car_mass * whl_radius / gear_ratio;
    constexpr float approx_propogation_delay = 0.02f;  // sec
    constexpr float approx_mppi_time = 0.02f; // sec

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