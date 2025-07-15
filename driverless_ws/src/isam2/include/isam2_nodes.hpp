/**
 * @file isam2Node.hpp
 * @author your name (you@domain.com)
 * @brief 
 * The SLAM nodes inherit from the Generic SLAM node, including between
 * RealDataSLAMNodeGPS and RealDataSLAMNodeNoGPS. The reason being is 
 * that each SLAM node must register the sync callback. 
 * 
 * You cannot just define sync to be a virtual variable because the sync
 * policy type is different for each of the nodes. If you really wanted 
 * to, you could define a polymorphic sync variable. 
 * @version 0.1
 * @date 2025-05-05
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#pragma once
#include "ros_utils.hpp"
#include "isam2.hpp"

namespace nodes {
    template<typename cone_msg_t, typename velocity_msg_t, typename orientation_msg_t, typename position_msg_t> 
    class GenericSLAMNode : public rclcpp::Node {
    public: 
        GenericSLAMNode();

    protected:
        slam::slamISAM slam_instance; /* We need to initialize this because it is used in the constructor */
        std::chrono::high_resolution_clock::time_point cur_sync_callback_time;
        std::optional<std::chrono::high_resolution_clock::time_point> prev_sync_callback_time;

        // rclcpp::Publisher<interfaces::msg::SLAMData>::SharedPtr slam_publisher_; 
        rclcpp::Publisher<interfaces::msg::SLAMPose>::SharedPtr slam_pose_publisher; 
        rclcpp::Publisher<interfaces::msg::SLAMChunk>::SharedPtr slam_chunk_publisher; 
        gtsam::Pose2 velocity;

        std::optional<gtsam::Point2> init_lon_lat; // local variable to load odom into SLAM instance when msg in lon lat
        std::optional<gtsam::Point2> init_x_y; // local variable to load odom into SLAM instance when msg in x and y meters

        gtsam::Point2 gps_position; // local variable to load odom into SLAM instance
        double yaw;

        std::vector<gtsam::Point2> cones; // local variable to load cone observations into SLAM instance
        std::vector<gtsam::Point2> orange_cones; // local variable to load cone observations into SLAM instance
        std::vector<gtsam::Point2> blue_cones; //local variable to store the blue observed cones
        std::vector<gtsam::Point2> yellow_cones; //local variable to store the yellow observed cones


        bool file_opened;

        rclcpp::TimerBase::SharedPtr timer;
        double dt;

        std::optional<std_msgs::msg::Header> prev_filter_time;

        //print files
        std::ofstream outfile;
        std::ofstream pose_cones;   

        /* QOS */
        const rmw_qos_profile_t reliable_profile = {
            RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            30,
            RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            RMW_QOS_POLICY_DURABILITY_VOLATILE,
            RMW_QOS_DEADLINE_DEFAULT,
            RMW_QOS_LIFESPAN_DEFAULT,
            RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
            RMW_QOS_LIVELINESS_LEASE_DURATION_DEFAULT,
            false
        };
        const rmw_qos_profile_t best_effort_profile = {
            RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            30,
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
               best_effort_profile.depth),
           best_effort_profile);

        message_filters::Subscriber<cone_msg_t> cone_sub;
        message_filters::Subscriber<velocity_msg_t> vehicle_vel_sub;
        message_filters::Subscriber<orientation_msg_t> vehicle_angle_sub;   
        message_filters::Subscriber<position_msg_t> vehicle_pos_sub;

        void declare_yaml_params();
        std::optional<yaml_params::NoiseInputs> get_noise_inputs();
        void publish_slam_data(const slam::slam_output_t& slam_data, std_msgs::msg::Header header);

        virtual void perform_subscribes() = 0;
    };

    

    /**
     * @brief Sometimes we run on rosbag data and have GPS
     * 
     */
    class RealDataSLAMNodeGPS : public GenericSLAMNode<interfaces::msg::ConeArray,
                                            geometry_msgs::msg::TwistStamped,
                                            geometry_msgs::msg::QuaternionStamped,
                                            geometry_msgs::msg::Vector3Stamped>
    {


        using cone_msg_t = interfaces::msg::ConeArray;
        using velocity_msg_t = geometry_msgs::msg::TwistStamped; 
        using orientation_msg_t = geometry_msgs::msg::QuaternionStamped;
        using position_msg_t = geometry_msgs::msg::Vector3Stamped;
        using sync_policy = message_filters::sync_policies::ApproximateTime<cone_msg_t, velocity_msg_t, orientation_msg_t, position_msg_t>;


        protected:
        /* Subscribers*/
        void perform_subscribes() override;

        /* Callbacks */
        void cone_callback(const typename cone_msg_t::ConstSharedPtr &cone_data);
        void vehicle_vel_callback(const typename velocity_msg_t::ConstSharedPtr &vehicle_vel_data);
        void vehicle_angle_callback(const typename orientation_msg_t::ConstSharedPtr &vehicle_angle_data);
        void vehicle_pos_callback(const typename position_msg_t::ConstSharedPtr &vehicle_pos_data);
        void sync_callback (const cone_msg_t::ConstSharedPtr &cone_data,
                const velocity_msg_t::ConstSharedPtr &vehicle_vel_data,
                const orientation_msg_t::ConstSharedPtr &vehicle_angle_data, 
                const position_msg_t::ConstSharedPtr &vehicle_pos_data);
        std::shared_ptr<message_filters::Synchronizer<sync_policy>> sync;


        public: 
        RealDataSLAMNodeGPS();
    };

    /**
     * @brief Sometimes we run on rosbag data and GPS
     * 
     */
    class RealDataSLAMNodeNoGPS : public GenericSLAMNode<interfaces::msg::ConeArray,
                                            geometry_msgs::msg::TwistStamped,
                                            geometry_msgs::msg::QuaternionStamped,
                                            geometry_msgs::msg::Vector3Stamped>
    {
        using cone_msg_t = interfaces::msg::ConeArray;
        using velocity_msg_t = geometry_msgs::msg::TwistStamped; 
        using orientation_msg_t = geometry_msgs::msg::QuaternionStamped;
        using dummy_position_msg_t = geometry_msgs::msg::Vector3Stamped;
        using sync_policy = message_filters::sync_policies::ApproximateTime<cone_msg_t, velocity_msg_t, orientation_msg_t>;


        protected:

        /* Subscribers*/
        void perform_subscribes() override;

        /* Callbacks */
        void cone_callback(const cone_msg_t::ConstSharedPtr &cone_data);
        void vehicle_vel_callback(const velocity_msg_t::ConstSharedPtr &vehicle_vel_data);
        void vehicle_angle_callback(const orientation_msg_t::ConstSharedPtr &vehicle_angle_data);
        void sync_callback (const cone_msg_t::ConstSharedPtr &cone_data,
                const velocity_msg_t::ConstSharedPtr &vehicle_vel_data,
                const orientation_msg_t::ConstSharedPtr &vehicle_angle_data);
        std::shared_ptr<message_filters::Synchronizer<sync_policy>> sync;


        public:
        RealDataSLAMNodeNoGPS();
    };

    /**
     * @brief In controls sim, the orientation information is incorporated into 
     * the PoseStamped message
     * 
     */
    class ControlsSimSLAMNode : public GenericSLAMNode <interfaces::msg::ConeArray,
                                                geometry_msgs::msg::TwistStamped, 
                                                geometry_msgs::msg::QuaternionStamped,
                                                geometry_msgs::msg::PoseStamped>
    {
        using cone_msg_t = interfaces::msg::ConeArray;
        using velocity_msg_t = geometry_msgs::msg::TwistStamped; 
        using dummy_orientation_msg_t = geometry_msgs::msg::QuaternionStamped;
        using pose_msg_t = geometry_msgs::msg::PoseStamped;

        using sync_policy = message_filters::sync_policies::ApproximateTime<cone_msg_t, velocity_msg_t, pose_msg_t>;

        protected:

        /* Subscribers */
        void perform_subscribes() override;

        /* Callbacks */
        void cone_callback(const cone_msg_t::ConstSharedPtr &cone_data);
        void vehicle_vel_callback(const velocity_msg_t::ConstSharedPtr &vehicle_vel_data);
        void vehicle_pos_callback(const pose_msg_t::ConstSharedPtr &vehicle_pos_data);
        void sync_callback (const cone_msg_t::ConstSharedPtr &cone_data,
                const velocity_msg_t::ConstSharedPtr &vehicle_vel_data,
                const pose_msg_t::ConstSharedPtr &vehicle_pos_data);
        std::shared_ptr<message_filters::Synchronizer<sync_policy>> sync;


        public:
        ControlsSimSLAMNode();
    };

}
