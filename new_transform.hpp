#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "geometry_msgs/msg/vector3.hpp"

/**
 * @brief Transforms a 3D point in LiDAR space to 2D pixel coordinates in camera space
 * 
 * @param point The 3D point from LiDAR
 * @param projection_matrix_l Projection matrix for left camera
 * @param projection_matrix_r Projection matrix for right camera
 * @return std::pair<Eigen::Vector2d, Eigen::Vector2d> Pixel coordinates in both cameras
 */
std::pair<Eigen::Vector3d, Eigen::Vector3d> transform_point(
    geometry_msgs::msg::Vector3& point,
    std::pair<std::pair<double, double>, std::pair<double, double>> ds_pair,
    std::pair<double, double> left_right_dyaw,
    const std::pair<Eigen::Matrix<double, 3, 4>, Eigen::Matrix<double, 3, 4>> &projection_matrix_pair
);

/**
 * @brief Takes a change in x and y in global frame and returns a change in x and y in local CMR frame
 *
 * @param global_frame_change The change in x and y in global frame
 * @param yaw yaw in radians
 * @return std::pair<double, double>
 */
std::pair<double, double> global_frame_to_local_frame(
    std::pair<double, double> global_frame_change,
    double yaw
);