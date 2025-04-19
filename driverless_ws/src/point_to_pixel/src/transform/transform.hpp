#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "rclcpp/rclcpp.hpp"

/**
 * @brief Transforms a 3D point in LiDAR space to 2D pixel coordinates in camera space
 *
 * @param point The 3D point from LiDAR
 * @param projection_matrix_l Projection matrix for left camera
 * @param projection_matrix_r Projection matrix for right camera
 * @return std::pair<Eigen::Vector2d, Eigen::Vector2d> Pixel coordinates in both cameras
 */
std::pair<Eigen::Vector3d, Eigen::Vector3d> transform_point(
    const rclcpp::Logger &logger,
    geometry_msgs::msg::Vector3 &point,
    std::pair<std::pair<double, double>, std::pair<double, double>> ds_pair,
    std::pair<double, double> left_right_dyaw,
    const std::pair<Eigen::Matrix<double, 3, 4>, Eigen::Matrix<double, 3, 4>> &projection_matrix_pair);

/**
 * @brief Takes a change in x and y in global frame and returns a change in x and y in local CMR frame
 *
 * @param global_frame_change The change in x and y in global frame
 * @param yaw yaw in radians
 * @return std::pair<double, double>
 */
std::pair<double, double> global_frame_to_local_frame(
    std::pair<double, double> global_frame_change,
    double yaw);

/**
 * @brief Get the velocity and yaw at the time of the frame
 *
 * @param logger Logger for logging messages
 * @param yaw_mutex Mutex for yaw deque
 * @param velocity_mutex Mutex for velocity deque
 * @param velocity_deque Deque of velocity messages
 * @param yaw_deque Deque of yaw messages
 * @param frameTime Time of the frame in nanoseconds
 * @return std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr>
 */
std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> get_velocity_yaw(
    const rclcpp::Logger &logger,
    std::mutex *yaw_mutex,
    std::mutex *velocity_mutex,
    const std::deque<geometry_msgs::msg::TwistStamped::SharedPtr> &velocity_deque,
    const std::deque<geometry_msgs::msg::Vector3Stamped::SharedPtr> &yaw_deque,
    uint64_t frameTime
);