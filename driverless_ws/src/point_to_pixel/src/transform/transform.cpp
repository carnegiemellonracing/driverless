#include "transform.hpp"
#include "rclcpp/rclcpp.hpp"

std::pair<double, double> global_frame_to_local_frame(
    std::pair<double, double> global_frame_change,
    double yaw)
{
    double global_frame_dx = global_frame_change.first;
    double global_frame_dy = global_frame_change.second;

    double cmr_y = global_frame_dx * std::cos(yaw * M_PI / 180.0) + global_frame_dy * std::sin(yaw * M_PI / 180.0);
    double cmr_x = global_frame_dx * std::sin(yaw * M_PI / 180.0) - global_frame_dy * std::cos(yaw * M_PI / 180.0);

    return std::make_pair(cmr_x, cmr_y);
}

/**
 * @brief
 *
 * @param local_frame_change
 * @param dyaw The change in yaw in radians
 * @return std::pair<double, double>
 */
std::pair<double, double> local_frame_to_local_frame(
    std::pair<double, double> local_frame_change,
    double dyaw)
{
    double local_frame_dx = local_frame_change.first;
    double local_frame_dy = local_frame_change.second;

    double cmr_y = -local_frame_dx * std::sin(dyaw * M_PI / 180.0) + local_frame_dy * std::cos(dyaw * M_PI / 180.0);
    double cmr_x = local_frame_dx * std::cos(dyaw * M_PI / 180.0) + local_frame_dy * std::sin(dyaw * M_PI / 180.0);

    return std::make_pair(cmr_x, cmr_y);
}

std::pair<double, double> motion_model_on_point(
    std::pair<double, double> dv_frame_change,
    double point_x,
    double point_y,
    double dyaw)
{
    // The change in x and y of the car's motion wrt to the normalized frame of the start position
    double local_car_dx = dv_frame_change.first;
    double local_car_dy = dv_frame_change.second;

    // Calculate the change in x and y from the cone to the second position, wrt to the normalized frame of the start position
    double cone_to_car_dx = point_x - local_car_dx;
    double cone_to_car_dy = point_y - local_car_dy;
    std::pair<double, double> cone_wrt_normalized_start_frame = std::make_pair(cone_to_car_dx, cone_to_car_dy);

    return local_frame_to_local_frame(cone_wrt_normalized_start_frame, dyaw);
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> transform_point(
    geometry_msgs::msg::Vector3 &point,
    std::pair<std::pair<double, double>, std::pair<double, double>> ds_pair,
    std::pair<double, double> left_right_dyaw,
    const std::pair<Eigen::Matrix<double, 3, 4>, Eigen::Matrix<double, 3, 4>> &projection_matrix_pair)
{
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
    std::pair<double, double> lidar_pt_l_xy = motion_model_on_point(ds_pair.first, point.x, point.y, left_right_dyaw.first);
    std::pair<double, double> lidar_pt_r_xy = motion_model_on_point(ds_pair.second, point.x, point.y, left_right_dyaw.second);
    Eigen::Vector4d lidar_pt_l(lidar_pt_l_xy.first, lidar_pt_l_xy.second, point.z, 1.0);
    Eigen::Vector4d lidar_pt_r(lidar_pt_r_xy.first, lidar_pt_r_xy.second, point.z, 1.0);

    double distance_l = std::sqrt(lidar_pt_l(0) * lidar_pt_l(0) + lidar_pt_l(1) * lidar_pt_l(1));
    double distance_r = std::sqrt(lidar_pt_r(0) * lidar_pt_r(0) + lidar_pt_r(1) * lidar_pt_r(1));

    // Apply projection matrix to LiDAR point
    Eigen::Vector3d transformed_l = projection_matrix_pair.first * lidar_pt_l;
    Eigen::Vector3d transformed_r = projection_matrix_pair.second * lidar_pt_r;

    // Divide by z coordinate for Euclidean normalization
    // Third field represents XY euclidean distance
    Eigen::Vector3d pixel_l(transformed_l(0) / transformed_l(2), transformed_l(1) / transformed_l(2), distance_l);
    Eigen::Vector3d pixel_r(transformed_r(0) / transformed_r(2), transformed_r(1) / transformed_r(2), distance_r);

    return std::make_pair(pixel_l, pixel_r);
}