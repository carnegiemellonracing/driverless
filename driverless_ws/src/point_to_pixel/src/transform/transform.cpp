#include "transform.hpp"

std::pair<Eigen::Vector3d, Eigen::Vector3d> transform_point(
    geometry_msgs::msg::Vector3& point,
    const Eigen::Matrix<double, 3, 4>& projection_matrix_l,
    const Eigen::Matrix<double, 3, 4>& projection_matrix_r
) {
    // Convert point to Eigen Vector4d (homogeneous coordinates)
    Eigen::Vector4d lidar_pt(point.x, point.y, point.z, 1.0);

    // Apply projection matrix to LiDAR point
    Eigen::Vector3d transformed_l = projection_matrix_l * lidar_pt;
    Eigen::Vector3d transformed_r = projection_matrix_r * lidar_pt;

    // Divide by z coordinate for Euclidean normalization
    // Include z coordinate for depth
    Eigen::Vector3d pixel_l(transformed_l(0)/transformed_l(2), transformed_l(1)/transformed_l(2), transformed_l(2));
    Eigen::Vector3d pixel_r(transformed_r(0)/transformed_r(2), transformed_r(1)/transformed_r(2), transformed_r(2));

    return std::make_pair(pixel_l, pixel_r);
} 