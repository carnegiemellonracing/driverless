#include "isam2_nodes.hpp"


namespace nodes {
    template<typename cone_msg_t, typename velocity_msg_t, typename orientation_msg_t, typename position_msg_t> 
    GenericSLAMNode<cone_msg_t, velocity_msg_t, orientation_msg_t, position_msg_t>::GenericSLAMNode() : Node("slam_node")
    {
        // slam_publisher_ = this->create_publisher<interfaces::msg::SLAMData>(SLAM_TOPIC, 10);
        slam_pose_publisher = this->create_publisher<interfaces::msg::SLAMPose>(SLAM_POSE_TOPIC, 10);
        slam_chunk_publisher = this->create_publisher<interfaces::msg::SLAMChunk>(SLAM_CHUNK_TOPIC, 10);

        declare_yaml_params();

        /* Update sync policy and subs when switching betw real data and sim*/ 
        dt = 0.0;
        std::optional<yaml_params::NoiseInputs> noise_inputs = get_noise_inputs();
        slam_instance = slam::slamISAM(this->get_logger(), noise_inputs);
        

        init_lon_lat = std::nullopt;
        init_x_y = std::nullopt;
        file_opened = true;

        prev_filter_time = std::nullopt;
        prev_sync_callback_time = std::nullopt;
        RCLCPP_INFO(this->get_logger(), "----- Reminder to subscribe to the right topics in constants.hpp -----");
    }

    template<typename cone_msg_t, typename velocity_msg_t, typename orientation_msg_t, typename position_msg_t> 
    void GenericSLAMNode<cone_msg_t, velocity_msg_t, orientation_msg_t, position_msg_t>::publish_slam_data(const slam::slam_output_t& slam_data, std_msgs::msg::Header header) {
        interfaces::msg::SLAMPose pose_msg = interfaces::msg::SLAMPose();
        pose_msg.header = header;
        pose_msg.current_chunk_id.data = 0;
        pose_msg.pose = std::get<2>(slam_data);

        interfaces::msg::SLAMChunk chunk_msg = interfaces::msg::SLAMChunk();
        chunk_msg.header = header;
        chunk_msg.blue_cones = std::get<0>(slam_data);
        chunk_msg.chunk_id.data = 0;
        chunk_msg.yellow_cones = std::get<1>(slam_data);

        slam_pose_publisher->publish(pose_msg);
        slam_chunk_publisher->publish(chunk_msg);
    }

    
    template<typename cone_msg_t, typename velocity_msg_t, typename orientation_msg_t, typename position_msg_t> 
    void GenericSLAMNode<cone_msg_t, velocity_msg_t, orientation_msg_t, position_msg_t>::declare_yaml_params() {
        this->declare_parameter<bool>("use_yaml", false);
        this->declare_parameter<double>("yaml_prior_imu_x_std_dev", IMU_X_STD_DEV);
        this->declare_parameter<double>("yaml_prior_imu_y_std_dev", IMU_Y_STD_DEV);
        this->declare_parameter<double>("yaml_prior_imu_heading_std_dev", IMU_HEADING_STD_DEV);

        this->declare_parameter<double>("yaml_bearing_std_dev", BEARING_STD_DEV);
        this->declare_parameter<double>("yaml_range_std_dev", RANGE_STD_DEV);

        this->declare_parameter<double>("yaml_imu_x_std_dev", IMU_X_STD_DEV);
        this->declare_parameter<double>("yaml_imu_y_std_dev", IMU_Y_STD_DEV);
        this->declare_parameter<double>("yaml_imu_heading_std_dev", IMU_HEADING_STD_DEV);
        this->declare_parameter<double>("yaml_gps_x_std_dev", GPS_X_STD_DEV);
        this->declare_parameter<double>("yaml_gps_y_std_dev", GPS_Y_STD_DEV);

        this->declare_parameter<int>("yaml_look_radius", static_cast<int>(LOOK_RADIUS)); // tell us how many cones back and forth to update in slam_est
        this->declare_parameter<int>("yaml_min_cones_update_all", static_cast<int>(MIN_CONES_UPDATE_ALL));
        this->declare_parameter<int>("yaml_window_update",  static_cast<int>(WINDOW_UPDATE));
        this->declare_parameter<int>("yaml_update_start_n", static_cast<int>(UPDATE_START_N)); // tell us how many cones back and forth to update in slam_est
        this->declare_parameter<int>("yaml_update_recent_n",static_cast<int>(UPDATE_RECENT_N));

        this->declare_parameter<double>("yaml_imu_offset", IMU_OFFSET); //meters; offset from the center of the car
        this->declare_parameter<double>("yaml_lidar_offset", LIDAR_OFFSET); //meters; offset from the center of the car
        this->declare_parameter<double>("yaml_max_cone_range", MAX_CONE_RANGE);
        this->declare_parameter<double>("yaml_turning_max_cone_range", TURNING_MAX_CONE_RANGE);
        this->declare_parameter<double>("yaml_dist_from_start_loop_closure_th", DIST_FROM_START_LOOP_CLOSURE_TH); //meters; distance from the start for loop closure detection
        this->declare_parameter<double>("yaml_m_dist_th", M_DIST_TH);
        this->declare_parameter<double>("yaml_turning_m_dist_th", TURNING_M_DIST_TH);
        this->declare_parameter<int>("yaml_update_iterations_n", static_cast<int>(UPDATE_ITERATIONS_N));
        this->declare_parameter<int>("yaml_return_n_cones", static_cast<int>(RETURN_N_CONES));
    }
    
    template<typename cone_msg_t, typename velocity_msg_t, typename orientation_msg_t, typename position_msg_t> 
    std::optional<yaml_params::NoiseInputs> GenericSLAMNode<cone_msg_t, velocity_msg_t, orientation_msg_t, position_msg_t>::get_noise_inputs() {
        std::optional<yaml_params::NoiseInputs> noise_inputs = std::nullopt;
        /* Note: get_parameter returns int64_t for ints. However, no parameter at the moment needs to be in64_t
         * so we static_cast to int32_t */

        RCLCPP_INFO(this->get_logger(), "--------has_parameter 'use_yaml': %s --------", (this->has_parameter("use_yaml") ? "true" : "false"));
        if (this->has_parameter("use_yaml")) {
            RCLCPP_INFO(this->get_logger(), "--------use_yaml: %s --------", (this->get_parameter("use_yaml").as_bool() ? "true" : "false"));
        }

        if (this->has_parameter("use_yaml") && this->get_parameter("use_yaml").as_bool()) {
            yaml_params::NoiseInputs noise_inputs_literal;
            noise_inputs_literal.yaml_prior_imu_x_std_dev = this->get_parameter("yaml_prior_imu_x_std_dev").as_double();
            noise_inputs_literal.yaml_prior_imu_y_std_dev = this->get_parameter("yaml_prior_imu_y_std_dev").as_double();
            noise_inputs_literal.yaml_prior_imu_heading_std_dev = this->get_parameter("yaml_prior_imu_heading_std_dev").as_double();

            noise_inputs_literal.yaml_bearing_std_dev = this->get_parameter("yaml_bearing_std_dev").as_double();
            noise_inputs_literal.yaml_range_std_dev = this->get_parameter("yaml_range_std_dev").as_double();

            noise_inputs_literal.yaml_imu_x_std_dev = this->get_parameter("yaml_imu_x_std_dev").as_double();
            noise_inputs_literal.yaml_imu_y_std_dev = this->get_parameter("yaml_imu_y_std_dev").as_double();
            noise_inputs_literal.yaml_imu_heading_std_dev = this->get_parameter("yaml_imu_heading_std_dev").as_double();

            noise_inputs_literal.yaml_gps_x_std_dev = this->get_parameter("yaml_gps_x_std_dev").as_double();
            noise_inputs_literal.yaml_gps_y_std_dev = this->get_parameter("yaml_gps_y_std_dev").as_double();

            noise_inputs_literal.yaml_look_radius = static_cast<std::size_t>(this->get_parameter("yaml_look_radius").as_int());
            noise_inputs_literal.yaml_min_cones_update_all = static_cast<std::size_t>(this->get_parameter("yaml_min_cones_update_all").as_int());
            noise_inputs_literal.yaml_window_update =   static_cast<std::size_t>(this->get_parameter("yaml_window_update").as_int());
            noise_inputs_literal.yaml_update_start_n =  static_cast<std::size_t>(this->get_parameter("yaml_update_start_n").as_int());
            noise_inputs_literal.yaml_update_recent_n = static_cast<std::size_t>(this->get_parameter("yaml_update_recent_n").as_int());

            noise_inputs_literal.yaml_imu_offset = this->get_parameter("yaml_imu_offset").as_double();
            noise_inputs_literal.yaml_lidar_offset = this->get_parameter("yaml_lidar_offset").as_double();

            noise_inputs_literal.yaml_max_cone_range = this->get_parameter("yaml_max_cone_range").as_double();
            noise_inputs_literal.yaml_turning_max_cone_range = this->get_parameter("yaml_turning_max_cone_range").as_double();
            noise_inputs_literal.yaml_dist_from_start_loop_closure_th = this->get_parameter("yaml_dist_from_start_loop_closure_th").as_double();

            noise_inputs_literal.yaml_m_dist_th = this->get_parameter("yaml_m_dist_th").as_double();
            noise_inputs_literal.yaml_turning_m_dist_th = this->get_parameter("yaml_turning_m_dist_th").as_double();
            noise_inputs_literal.yaml_update_iterations_n = static_cast<std::size_t>(this->get_parameter("yaml_update_iterations_n").as_int());
            
            noise_inputs_literal.yaml_return_n_cones = static_cast<std::size_t>(this->get_parameter("yaml_return_n_cones").as_int());
            noise_inputs = noise_inputs_literal;
                
        } else {
            RCLCPP_INFO(this->get_logger(), "--------No yaml file found --------");
        }
        return noise_inputs;
    }

    


}




