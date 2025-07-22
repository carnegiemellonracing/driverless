#pragma once

#include <memory>
#include <optional>

#include "rclcpp/rclcpp.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "interfaces/msg/cone_array.hpp"
#include "interfaces/msg/cone_array_with_odom.hpp"
// #include "interfaces/msg/slam_data.hpp"
#include "interfaces/msg/slam_pose.hpp"
#include "interfaces/msg/slam_chunk.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/quaternion_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "sensor_msgs/msg/nav_sat_fix.hpp"
#include "std_msgs/msg/string.hpp"

#include <boost/shared_ptr.hpp>
#include <vector>
#include <fstream>
#include <deque>
#include <cmath>
#include <chrono>
#include <climits>
#include <fmt/format.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>
#include "constants.hpp"


/**
 * @brief Defines conversion functions for converting between types when 
 * subscribing to or publishing messages.
 * 
 */
namespace ros_msg_conversions {
    void cone_msg_to_vectors(const interfaces::msg::ConeArray::ConstSharedPtr &cone_data,
                                            std::vector<gtsam::Point2> &cones,
                                            std::vector<gtsam::Point2> &blue_cones,
                                            std::vector<gtsam::Point2> &yellow_cones,
                                            std::vector<gtsam::Point2> &orange_cones);

    gtsam::Pose2 velocity_msg_to_pose2(const geometry_msgs::msg::TwistStamped::ConstSharedPtr &vehicle_vel_data);

    gtsam::Pose2 posestamped_msg_to_pose2(const geometry_msgs::msg::PoseStamped::ConstSharedPtr &vehicle_pos_data, gtsam::Point2 init_x_y, rclcpp::Logger logger);

    double quat_msg_to_yaw(const geometry_msgs::msg::QuaternionStamped::ConstSharedPtr &vehicle_angle_data);

    gtsam::Point2 vector3_msg_to_gps(const geometry_msgs::msg::Vector3Stamped::ConstSharedPtr &vehicle_pos_data, gtsam::Point2 init_lon_lat, rclcpp::Logger logger);

    geometry_msgs::msg::Point point2_to_geometry_msg (gtsam::Point2 gtsam_point);
    std::vector<geometry_msgs::msg::Point> slam_est_to_points (std::vector<gtsam::Point2> gtsam_points, gtsam::Pose2 pose);
}

namespace motion_modeling {
    void imu_axes_to_DV_axes(double &x, double &y);
    gtsam::Point2 calc_lateral_velocity_error(double ang_velocity, double yaw);

    std::pair<gtsam::Pose2, gtsam::Pose2> velocity_motion_model(gtsam::Pose2 velocity, double dt,gtsam::Pose2 prev_pose, double yaw);

    gtsam::Point2 calc_offset_imu_to_car_center(double yaw);
    gtsam::Point2 calc_offset_lidar_to_car_center(double yaw);

    /**
     * @brief 
     * 
     * @param odometry 
     * @param velocity 
     * @param dt 
     * @param prev_pose 
     * @param global_odom 
     * @return std::pair<gtsam::Pose2, gtsam::Pose2> Returns a pair where the first element is the new_pose and the 
     * second element is the odometry. 
     */
    std::pair<gtsam::Pose2, gtsam::Pose2> gps_motion_model(gtsam::Pose2 prev_pose, gtsam::Pose2 global_odom);

    /**
     * @brief Returns a pair of bools where the first element is true if the 
     * car is moving, and the second element is true if the car is turning.
     * 
     * @param velocity 
     * @return std::pair<bool, bool> 
     */
    std::pair<bool, bool> determine_movement(gtsam::Pose2 velocity);
    double header_to_nanosec(const std_msgs::msg::Header &header);
    double header_to_dt(std::optional<std_msgs::msg::Header> prev, std::optional<std_msgs::msg::Header> cur);
    double degrees_to_radians(double degrees);
}

namespace yaml_params {
    struct NoiseInputs {
        double yaml_prior_imu_x_std_dev;
        double yaml_prior_imu_y_std_dev;
        double yaml_prior_imu_heading_std_dev;

        double yaml_bearing_std_dev;
        double yaml_range_std_dev;

        double yaml_imu_x_std_dev;
        double yaml_imu_y_std_dev;
        double yaml_imu_heading_std_dev;
        double yaml_gps_x_std_dev;
        double yaml_gps_y_std_dev;

        int yaml_look_radius;
        int yaml_min_cones_update_all;
        int yaml_window_update;
        int yaml_update_start_n;
        int yaml_update_recent_n;

        double yaml_imu_offset;
        double yaml_lidar_offset;
        double yaml_max_cone_range;
        double yaml_turning_max_cone_range;

        double yaml_dist_from_start_loop_closure_th;
        double yaml_m_dist_th;
        double yaml_turning_m_dist_th;
        int yaml_update_iterations_n;

        int yaml_return_n_cones;
    };
}


namespace cone_utils {
    Eigen::MatrixXd calc_cone_range_from_car(const std::vector<gtsam::Point2> &cone_obs);
    Eigen::MatrixXd calc_cone_bearing_from_car(const std::vector<gtsam::Point2> &cone_obs);

    /**
     * @brief Removes far away observed cones. Observed cones that are far away are more erroneous
     * 
     * @param cone_obs The observed cones
     * @param threshold The threshold distance from the car
     */
    std::vector<gtsam::Point2> remove_far_cones(std::vector<gtsam::Point2> cone_obs, double threshold);

    std::vector<gtsam::Point2> local_to_global_frame( std::vector<gtsam::Point2> cone_obs, gtsam::Pose2 cur_pose);
    std::vector<gtsam::Point2> global_to_local_frame(std::vector<gtsam::Point2> cone_obs, gtsam::Pose2 cur_pose);

}

namespace logging_utils {
    void print_cone_obs(const std::vector<gtsam::Point2> &cone_obs, const std::string& cone_color, std::optional<rclcpp::Logger> logger);

    void print_step_input(std::optional<rclcpp::Logger> logger, 
        std::optional<gtsam::Point2> gps_opt, 
        double yaw,
        const std::vector<gtsam::Point2> &cone_obs_blue, 
        const std::vector<gtsam::Point2> &cone_obs_yellow,
        const std::vector<gtsam::Point2> &orange_ref_cones, 
        gtsam::Pose2 velocity, 
        double dt);

    void print_update_poses(gtsam::Pose2 &prev_pose, gtsam::Pose2 &new_pose, gtsam::Pose2 &odometry, gtsam::Pose2 &imu_offset_global_odom, std::optional<rclcpp::Logger> logger);


    void record_step_inputs(std::optional<rclcpp::Logger> logger, 
        std::optional<gtsam::Point2> gps_opt,
        double yaw,
        const std::vector<gtsam::Point2> &cone_obs_blue, 
        const std::vector<gtsam::Point2> &cone_obs_yellow,
        const std::vector<gtsam::Point2> &orange_ref_cones, 
        gtsam::Pose2 velocity, 
        double dt);

    void log_string (std::optional<rclcpp::Logger> logger, std::string input_string, bool flag);

}