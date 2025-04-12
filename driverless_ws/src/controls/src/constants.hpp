///@file

#pragma once

#include <rclcpp/rclcpp.hpp>
// Note: these header files are part of the ROS2 standard libraries
#include <ros_types_and_constants.hpp>
//TODO: these should all be inline constexpr (not currently broken because not ODR-used)

namespace controls {

    // Testing stuff

    constexpr bool send_to_can = true;
    constexpr bool ingest_midline = false;
    constexpr bool follow_midline_only = true;
    enum class StateProjectionMode {
        MODEL_MULTISET,
        NAIVE_SPEED_ONLY,
        POSITIONLLA_YAW_SPEED
    };
    constexpr bool testing_on_breezway = false;
    constexpr bool testing_on_rosbag = false; // so that even if we are not using model multiset, we can record the IRL data for posterity
    // also note that testing_on_rosbag true means we don't publish control actions anymore, is that alright?
    constexpr bool republish_perc_cones = true; // no harm in doing this besides latency
    constexpr bool publish_spline = true;

    // Timing flags
    constexpr bool log_render_and_sync_timing = false;
    constexpr bool log_state_projection_history = true;

    constexpr StateProjectionMode state_projection_mode = StateProjectionMode::MODEL_MULTISET;
    constexpr float maximum_speed_ms = 2.0f;
    constexpr float whl_radius = 0.215f;
    constexpr float gear_ratio = 14.0f;

    // This is for reference only
    constexpr uint16_t can_max_velocity_rpm = static_cast<uint16_t>((maximum_speed_ms * 60.0f * gear_ratio) / (2 * M_PI * whl_radius));
    
    
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
    constexpr float above_speed_threshold_cost = 1000.0f;

    constexpr float torque_1Nps_cost = 0.0f;
    constexpr float swangle_1radps_cost = 0.0f;
        // DEPRECATED
        constexpr float offset_1m_cost = 10.0f; ///< Cost for being 1m away from midline DEPRECATED
    constexpr float target_speed = 2.0f; ///< Linear cost for under target speed, NO cost for above, in m/s
    constexpr float speed_off_1mps_cost = 1.0f; ///< Cost for being 1m/s below target_speed

    // Cost params
    constexpr float progress_cost_multiplier = 0.6f;
    /// Reason for not using infinity: reduction uses log of the cost (trading precision for representable range).
    /// This covers the edge case where every trajectory goes out of bounds, allowing us to still extract useful information.
    constexpr float out_of_bounds_cost = 100.0f; ///< Cost for being out of (fake) track bound as defined by @ref track_width.

    // Midline/SVM
    constexpr float mesh_grid_spacing = 0.1f; //m
    constexpr float max_spline_length = 200.0f;
    constexpr int cone_augmentation_angle = 60;

    constexpr float lookahead_behind_squared = 25.0f;

    // AIM communication stuff
    constexpr int aim_signal_period_ms = 98;
    constexpr int can_swangle_listener_period_ms = 10;
    constexpr float default_p = 1.5f;
    constexpr float default_feedforward = 30.0f;


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

    constexpr float cg_to_side = 0.75f; // ACTUAL .75
    constexpr float cg_to_nose = 2.025f;

#ifdef USESYSID
    constexpr float cg_to_front = 0.83855;
    constexpr float cg_to_rear = 0.71145; // Also rear of car
    constexpr float car_mass = 245.0f;


#else
    
    // Car params
    //cg_to_front and cg_to_rear are from center of gravity to wheel base
    //cg_to_nose is actual front of car
    //Wheel base = 1.55, car length = 2.80
    constexpr float cg_to_front = 0.775; 
    constexpr float cg_to_rear = 0.775; //Also rear of car
    //constexpr float whl_base = 2.0f;
    /// gear ratio = motor speed / wheel speed = wheel torque / motor torque
    constexpr float car_mass = 210.0f;

#endif

    // NEW MODEL STUFF (the 5 tuned parameters)

    constexpr float rolling_drag_constant_kN = 0.147287f;
    constexpr float rolling_drag_linear = 12.604f;
    constexpr float rolling_drag_squared = 0.0f;
    constexpr float understeer_slope_squared = 0.28980572;
    constexpr float torque_efficiency = 0.75152479f;

    // OLD MODEL STUFF 
    constexpr float rolling_drag = 100.0f;   /// Total drag forces on the car (rolling friction + air drag) in N
    constexpr float understeer_slope = 0.0f; ///< How much car understeers as speed increases. See @rst :doc:`/source/explainers/slipless_model` @endrst.

    /// Maximum forward acceleration in m/s^2. Can be an imposed limit or the actual physics limitation.
    constexpr float long_tractive_capability = 2.0f;
    /// Maximum centripetal acceleration in m/s^2. Can be an imposed limit or the actual physics limitation.
    /// Usually slightly more than @c long_tractive_capability
    constexpr float lat_tractive_capability = 3.0f;
    constexpr float brake_enable_speed = 1.0f;
    /// Maximum torque request (N m)
    constexpr float saturating_motor_torque = (long_tractive_capability + rolling_drag / car_mass) * car_mass * whl_radius / gear_ratio;
    constexpr float min_torque = -saturating_motor_torque;
    constexpr float max_torque = saturating_motor_torque;
    constexpr float min_swangle = -19 * M_PI / 180.0f; //19 radians
    constexpr float max_swangle = 19 * M_PI / 180.0f;
    /// Time from MPPI control action request to physical change, in sec
    // TODO: Re-estimate since Falcon (steering motor) replacement
    constexpr float approx_propogation_delay = 0.0f;
    constexpr float approx_mppi_time = 0.020f; ///< Time from MPPI launch to control action calculation, in sec

    enum class TorqueMode
    {
        AWD,
        FWD,
        RWD
    };
    constexpr TorqueMode torque_mode = TorqueMode::AWD; ///< Because back two wheels don't currently work


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