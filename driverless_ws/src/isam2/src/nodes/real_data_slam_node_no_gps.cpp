#include "isam2_nodes.hpp"

namespace nodes {
    /*=======================================================================
    *                          RealDataSLAMNodeNoGPS
    *=======================================================================*/

    RealDataSLAMNodeNoGPS::RealDataSLAMNodeNoGPS() : GenericSLAMNode<RealDataSLAMNodeNoGPS::cone_msg_t,
                                                                    RealDataSLAMNodeNoGPS::velocity_msg_t,
                                                                    RealDataSLAMNodeNoGPS::orientation_msg_t,
                                                                    RealDataSLAMNodeNoGPS::dummy_position_msg_t>()
    {
        perform_subscribes();
        sync = std::make_shared<message_filters::Synchronizer<RealDataSLAMNodeNoGPS::sync_policy>>(
                                    RealDataSLAMNodeNoGPS::sync_policy(100),
                                    cone_sub, 
                                    vehicle_vel_sub, 
                                    vehicle_angle_sub);
        sync->setAgePenalty(0.1);
        sync->registerCallback(std::bind(&RealDataSLAMNodeNoGPS::sync_callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    }

    void RealDataSLAMNodeNoGPS::cone_callback(const RealDataSLAMNodeNoGPS::cone_msg_t::ConstSharedPtr &cone_data)  {
        auto cone_callback_start = std::chrono::high_resolution_clock::now();
        cones = {};
        blue_cones = {};
        yellow_cones = {};
        orange_cones = {};

        /* Process cones */
        ros_msg_conversions::cone_msg_to_vectors(cone_data, cones, blue_cones, yellow_cones, orange_cones);

        /* Timers */
        auto cone_callback_end = std::chrono::high_resolution_clock::now();
        auto cone_callback_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cone_callback_end - cone_callback_start);
        RCLCPP_INFO(this->get_logger(), "\tCone callback time: %ld", cone_callback_duration.count());
    }

    void RealDataSLAMNodeNoGPS::vehicle_vel_callback(const RealDataSLAMNodeNoGPS::velocity_msg_t::ConstSharedPtr &vehicle_vel_data)  {
        auto vehicle_vel_callback_start = std::chrono::high_resolution_clock::now();
        velocity = ros_msg_conversions::velocity_msg_to_pose2(vehicle_vel_data);

        /* Timers*/
        auto vehicle_vel_callback_end = std::chrono::high_resolution_clock::now();
        auto vehicle_vel_callback_duration = std::chrono::duration_cast<std::chrono::milliseconds>(vehicle_vel_callback_end - vehicle_vel_callback_start);
        RCLCPP_INFO(this->get_logger(), "\tVelocity callback time: %ld", vehicle_vel_callback_duration.count());
    }


    void RealDataSLAMNodeNoGPS::vehicle_angle_callback(const RealDataSLAMNodeNoGPS::orientation_msg_t::ConstSharedPtr &vehicle_angle_data)  {
        auto vehicle_angle_callback_start = std::chrono::high_resolution_clock::now();
        yaw = ros_msg_conversions::quat_msg_to_yaw(vehicle_angle_data);

        /* Timers*/
        auto vehicle_angle_callback_end = std::chrono::high_resolution_clock::now();
        auto vehicle_angle_callback_duration = std::chrono::duration_cast<std::chrono::milliseconds>(vehicle_angle_callback_end - vehicle_angle_callback_start);
        RCLCPP_INFO(this->get_logger(), "\tAngle callback time: %ld", vehicle_angle_callback_duration.count());
    }


    void RealDataSLAMNodeNoGPS::sync_callback(
        const RealDataSLAMNodeNoGPS::cone_msg_t::ConstSharedPtr &cone_data,
        const RealDataSLAMNodeNoGPS::velocity_msg_t::ConstSharedPtr &vehicle_vel_data,
        const RealDataSLAMNodeNoGPS::orientation_msg_t::ConstSharedPtr &vehicle_angle_data
    )  {
        // RCLCPP_INFO(this->get_logger(), "--------Start of Sync Callback--------");
        
        /* Getting the time between sync callbacks */
        cur_sync_callback_time = std::chrono::high_resolution_clock::now();
        if (prev_sync_callback_time.has_value()) {
            auto time_betw_sync_callbacks = std::chrono::duration_cast<std::chrono::milliseconds>(cur_sync_callback_time - prev_sync_callback_time.value());
            RCLCPP_INFO(this->get_logger(), "\tTime between sync_callbacks: %ld", time_betw_sync_callbacks.count());
        }
        


        auto sync_data_start = std::chrono::high_resolution_clock::now();
        std::optional<std_msgs::msg::Header> cur_filter_time(vehicle_vel_data->header);
        if (!prev_filter_time.has_value()) {
            prev_filter_time.swap(cur_filter_time);
            return;
        }

        dt = motion_modeling::header_to_dt(prev_filter_time, cur_filter_time);
        prev_filter_time.swap(cur_filter_time);

        /* Cone callback */
        RealDataSLAMNodeNoGPS::cone_callback(cone_data);
        /* Vehicle velocity callback */
        RealDataSLAMNodeNoGPS::vehicle_vel_callback(vehicle_vel_data);
        /* Vehicle angle callback */
        RealDataSLAMNodeNoGPS::vehicle_angle_callback(vehicle_angle_data);
        

        if (init_lon_lat.has_value()) {
            RCLCPP_INFO(this->get_logger(), "init_lon_lat: x:%f | y:%f\n", init_lon_lat.value().x(), init_lon_lat.value().y());
        }

        auto sync_data_end = std::chrono::high_resolution_clock::now();
        auto sync_data_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sync_data_end - sync_data_start);
        RCLCPP_INFO(this->get_logger(), "\tSync callback time: %ld \n", sync_data_duration.count());


        slam::slam_output_t slam_data = slam_instance.step(std::nullopt, yaw, blue_cones, yellow_cones, orange_cones, velocity, dt);
        publish_slam_data(slam_data, cone_data->header);

        // RCLCPP_INFO(this->get_logger(), "--------End of Sync Callback--------\n\n");
        prev_sync_callback_time.emplace(std::chrono::high_resolution_clock::now());
    }

    void RealDataSLAMNodeNoGPS::perform_subscribes()  {
        cone_sub.subscribe(this, CONE_DATA_TOPIC, best_effort_profile);
        vehicle_angle_sub.subscribe(this, VEHICLE_ANGLE_TOPIC, best_effort_profile);
        vehicle_vel_sub.subscribe(this, VEHICLE_VEL_TOPIC, best_effort_profile);
    }
}

int main(int argc, char* argv[]){

  rclcpp::init(argc, argv);
  rclcpp::spin(std::static_pointer_cast<rclcpp::Node>(std::make_shared<nodes::RealDataSLAMNodeNoGPS>()));
  rclcpp::shutdown();

  return 0;
}