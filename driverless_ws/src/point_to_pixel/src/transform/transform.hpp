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
    const Eigen::Matrix<double, 3, 4>& projection_matrix_l,
    const Eigen::Matrix<double, 3, 4>& projection_matrix_r
);