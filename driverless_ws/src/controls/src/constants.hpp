///@file

#pragma once

#include <rclcpp/rclcpp.hpp>

//TODO: these should all be inline constexpr (not currently broken because not ODR-used)

namespace controls { 
    /* ROS moments */

    constexpr const char *controller_node_name = "controller";
    constexpr const char *control_action_topic_name = "control_action";
    constexpr const char *spline_topic_name = "spline";
    constexpr const char *state_topic_name = "state";
    constexpr const char *cone_topic_name = "perc_cones"; //Is this right? didn't exist before
    constexpr const char *world_twist_topic_name = "filter/twist";
    constexpr const char *world_quat_topic_name = "filter/quaternion";
    constexpr const char *world_pose_topic_name = "filter/pose";
    constexpr const char *controller_info_topic_name = "controller_info";
    constexpr const char *pid_topic_name = "pid_values";
    constexpr const char *world_positionlla_topic_name = "filter/positionlla";

    // TODO: Ask Ankit what is this, why did we choose it
    /// Profile for best effort communication
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
    const rclcpp::QoS pid_qos (rclcpp::KeepLast(1));

    constexpr rcl_clock_type_t default_clock_type = RCL_ROS_TIME;

    // Testing stuff

    constexpr bool ingest_midline = false;
    constexpr bool follow_midline_only = false;
    enum class StateProjectionMode {
        MODEL_MULTISET,
        NAIVE_SPEED_ONLY,
        POSITIONLLA_YAW_SPEED
    };

    constexpr StateProjectionMode projection_mode = StateProjectionMode::NAIVE_SPEED_ONLY;
    constexpr uint16_t can_max_velocity_rpm = 3000;
    
    // Printing flags
    constexpr bool print_svm_timing = false;

    // MPPI stuff

    constexpr double controller_freq = 10.; ///< Target number of controller steps per second, in Hz. 1 controller step outputs 1 control action.
    constexpr float controller_period = 1. / controller_freq; ///< Target duration between control actions, in sec

    constexpr uint32_t num_samples = 64 * 1024; ///< Number of trajectories sampled each controller step
    constexpr uint32_t num_timesteps = 16; ///< Number of controller steps simulated into the future
    constexpr uint8_t action_dims = 2; ///< \f$q\f$, dimensions of @ref Action
    constexpr uint8_t state_dims = 4; ///< \f$p\f$, dimensions of @ref State
    constexpr float temperature = 1.0f; ///< Convergence speed/stability tradeoff, see LaTeX for more details
    constexpr unsigned long long seed = 0; ///< For brownian pseudo-RNG.
    /// Number of elements in the tensor containing all the sampled action trajectories.
    constexpr uint32_t num_action_trajectories = action_dims * num_timesteps * num_samples;
    /// Best guess of action trajectory when controller first starts.
    constexpr float init_action_trajectory[num_timesteps * action_dims] = {};
    constexpr float action_momentum = 0.0f; ///< How much of last action taken to retain in calculation of next action.

    // DEPRECATED
    constexpr float offset_1m_cost = 10.0f; ///< Cost for being 1m away from midline DEPRECATED
    constexpr float target_speed = 10.0f; ///< Linear cost for under target speed, NO cost for above, in m/s
    constexpr float speed_off_1mps_cost = 1.0f; ///< Cost for being 1m/s below target_speed

    // Cost params
    constexpr float progress_cost_multiplier = 0.6f;
    /// Reason for not using infinity: reduction uses log of the cost (trading precision for representable range).
    /// This covers the edge case where every trajectory goes out of bounds, allowing us to still extract useful information.
    constexpr float out_of_bounds_cost = 100.0f; ///< Cost for being out of (fake) track bound as defined by @ref track_width.
    // TODO: use real bounds

    // Midline/SVM
    constexpr float mesh_grid_spacing = 0.2f; //m
    constexpr float max_spline_length = 200.0f;
    constexpr int cone_augmentation_angle = 180;

    constexpr float lookahead_behind_squared = 25.0f;

    // AIM communication stuff
    constexpr int aim_signal_period_ms = 98;


    // State Estimation

    constexpr float spline_frame_separation = 0.5f;  // meters
    constexpr uint32_t curv_frame_lookup_tex_width = 512;
    constexpr float curv_frame_lookup_padding = 0; // meters
    /// Not real track width, used for curvilinear frame lookup table generation
    constexpr float fake_track_width = 10.0f;
    // mppi simulates a lot of shitty trajectories (naive brownian guess)
    /// Represents space the car occupies, used to calculate the size of the curvilinear lookup table.
    constexpr float car_padding = std::max(spline_frame_separation, M_SQRT2f32 * fake_track_width);
    constexpr bool reset_pose_on_cone = true; ///< Sets pose to 0 vector for new cone (sensor POV)
     // triangle threshold is the max distance between cones on opposing sides that we will use for triangle drawing
    constexpr float triangle_threshold_squared = 64.0f;

    // Car params
    //cg_to_front and cg_to_rear are from center of gravity to wheel base
    //cg_to_nose is actual front of car
    //Wheel base = 1.55, car length = 2.80
    constexpr float cg_to_front = 0.775; 
    constexpr float cg_to_rear = 0.775; //Also rear of car
    constexpr float cg_to_nose = 2.025f;
    constexpr float cg_to_side = 0.75f; //ACTUAL .75
    //constexpr float whl_base = 2.0f;
    constexpr float whl_radius = 0.2286;
    /// gear ratio = motor speed / wheel speed = wheel torque / motor torque
    constexpr float gear_ratio = 15.0f;
    constexpr float car_mass = 210.0f;
    constexpr float rolling_drag = 100.0f; /// Total drag forces on the car (rolling friction + air drag) in N
    /// Maximum forward acceleration in m/s^2. Can be an imposed limit or the actual physics limitation.
    constexpr float long_tractive_capability = 2.0f;
    /// Maximum centripetal acceleration in m/s^2. Can be an imposed limit or the actual physics limitation.
    /// Usually slightly more than @c long_tractive_capability
    constexpr float lat_tractive_capability = 3.0f;
    constexpr float understeer_slope = 0.0f; ///< How much car understeers as speed increases. See @rst :doc:`/source/explainers/slipless_model` @endrst.
    constexpr float brake_enable_speed = 1.0f;
    /// Maximum torque request (N m)
    constexpr float saturating_motor_torque = (long_tractive_capability + rolling_drag / car_mass) * car_mass * whl_radius / gear_ratio;
    /// Time from MPPI control action request to physical change, in sec
    // TODO: Re-estimate since Falcon (steering motor) replacement
    constexpr float approx_propogation_delay = 0.02f;
    constexpr float approx_mppi_time = 0.040f; ///< Time from MPPI launch to control action calculation, in sec

    enum class TorqueMode
    {
        AWD,
        FWD,
        RWD
    };
    constexpr TorqueMode torque_mode = TorqueMode::FWD; ///< Because back two wheels don't currently work


    // Indices

    constexpr uint8_t state_x_idx = 0;
    constexpr uint8_t state_y_idx = 1;
    constexpr uint8_t state_yaw_idx = 2;
    constexpr uint8_t state_speed_idx = 3;

    constexpr uint8_t action_swangle_idx = 0;
    constexpr uint8_t action_torque_idx = 1;


    // misc

    constexpr const char* clear_term_sequence = "\033c"; //TODO: wtf is this
}