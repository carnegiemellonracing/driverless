#include "transform.hpp"
#include "rclcpp/rclcpp.hpp"

std::pair<Eigen::Vector3d, Eigen::Vector3d> transform_point(
    geometry_msgs::msg::Vector3& point,
    const Eigen::Matrix<double, 3, 4>& projection_matrix_l,
    const Eigen::Matrix<double, 3, 4>& projection_matrix_r
) {
    std::stringstream ss_l;
    std::stringstream ss_r;

    // // Iterate over the rows and columns of the matrix and format the output
    // for (int i = 0; i < projection_matrix_l.rows(); ++i){
    //     for (int j = 0; j < projection_matrix_l.cols(); ++j){
    //         ss_l << projection_matrix_l(i, j) << " ";
    //         ss_r << projection_matrix_r(i, j) << " ";
    //     }
    //     ss_l << "\n";
    //     ss_r << "\n";
    // }
    // // Log the projection_matrix using ROS 2 logger
    // std::cout << "Projection Matrix Left:\n" << ss_l.str().c_str() << std::endl;
    // std::cout << "Projection Matrix Right:\n" << ss_r.str().c_str() << std::endl;

    // Convert point to Eigen Vector4d (homogeneous coordinates)
    Eigen::Vector4d lidar_pt(point.x, point.y, point.z, 1.0);

    // Apply projection matrix to LiDAR point
    Eigen::Vector3d transformed_l = projection_matrix_l * lidar_pt;
    Eigen::Vector3d transformed_r = projection_matrix_r * lidar_pt;

    // Divide by z coordinate for Euclidean normalization
    // Include z coordinate for depth
    Eigen::Vector3d pixel_l(transformed_l(0)/transformed_l(2), transformed_l(1)/transformed_l(2), transformed_l(2));
    Eigen::Vector3d pixel_r(transformed_r(0)/transformed_r(2), transformed_r(1)/transformed_r(2), transformed_r(2));

    std::cout << pixel_l(0) << " " << pixel_l(1) << std::endl;

    return std::make_pair(pixel_l, pixel_r);
} 