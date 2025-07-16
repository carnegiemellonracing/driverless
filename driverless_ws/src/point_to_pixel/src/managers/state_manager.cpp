#include "state_manager.hpp"
#include <stdexcept>

namespace point_to_pixel {
    std::pair<double, double> StateManager::global_frame_to_local_frame(
        std::pair<double, double> global_frame_change,
        double yaw)
    {
        double global_frame_dx = global_frame_change.first;
        double global_frame_dy = global_frame_change.second;

        double cmr_y = global_frame_dx * std::cos(yaw * M_PI / 180.0) + global_frame_dy * std::sin(yaw * M_PI / 180.0);
        double cmr_x = global_frame_dx * std::sin(yaw * M_PI / 180.0) - global_frame_dy * std::cos(yaw * M_PI / 180.0);

        return std::make_pair(cmr_x, cmr_y);
    }

    std::pair<Eigen::Vector3d, Eigen::Vector3d> StateManager::transform_point(
        const rclcpp::Logger &logger,
        geometry_msgs::msg::Vector3 &point,
        int64_t left_frame_time,
        int64_t right_frame_time,
        int64_t lidar_frame_time,
        const std::pair<Eigen::Matrix<double, 3, 4>, Eigen::Matrix<double, 3, 4>> &projection_matrix_pair)
    {

        // Motion modeling for both frames
        auto [vel_l_camera_frame, yaw_l_camera_frame] = get_prev_state(logger, left_frame_time);
        auto [vel_r_camera_frame, yaw_r_camera_frame] = get_prev_state(logger, right_frame_time);
        auto [vel_lidar_frame, yaw_lidar_frame] = get_prev_state(logger, lidar_frame_time);

        if (vel_lidar_frame == nullptr || yaw_lidar_frame == nullptr || 
            vel_l_camera_frame == nullptr || yaw_l_camera_frame == nullptr ||
            vel_r_camera_frame == nullptr || yaw_r_camera_frame == nullptr)
        {
            throw std::runtime_error("Could not get velocity and yaw; probably not recieving imu data");
        }

        double dt_l = (left_frame_time > lidar_frame_time) ? (left_frame_time - lidar_frame_time) / 1e9 : 0.0;
        double dt_r = (right_frame_time > lidar_frame_time) ? (right_frame_time - lidar_frame_time) / 1e9 : 0.0;

        double avg_vel_l_x = (vel_l_camera_frame->twist.linear.x + vel_lidar_frame->twist.linear.x) / 2;
        double avg_vel_l_y = (vel_l_camera_frame->twist.linear.y + vel_lidar_frame->twist.linear.y) / 2;
        double avg_vel_r_x = (vel_r_camera_frame->twist.linear.x + vel_lidar_frame->twist.linear.x) / 2;
        double avg_vel_r_y = (vel_r_camera_frame->twist.linear.y + vel_lidar_frame->twist.linear.y) / 2;

        // compute global position change
        auto global_dx_l = avg_vel_l_x * dt_l;
        auto global_dy_l = avg_vel_l_y * dt_l;
        auto global_dx_r = avg_vel_r_x * dt_r;
        auto global_dy_r = avg_vel_r_y * dt_r;

        double yaw_lidar_rad = yaw_lidar_frame->vector.z * M_PI / 180;
        std::pair<double, double> global_frame_change_l = {global_dx_l, global_dy_l};
        std::pair<double, double> global_frame_change_r = {global_dx_r, global_dy_r};
        std::pair<double, double> local_dyaw_l = global_frame_to_local_frame(global_frame_change_l, yaw_lidar_rad);
        std::pair<double, double> local_dyaw_r = global_frame_to_local_frame(global_frame_change_r, yaw_lidar_rad);

        auto global_dyaw_l = yaw_l_camera_frame->vector.z - yaw_lidar_frame->vector.z;
        auto global_dyaw_r = yaw_r_camera_frame->vector.z - yaw_lidar_frame->vector.z;

        // Convert point to Eigen Vector4d (homogeneous coordinates)
        std::pair<double, double> lidar_pt_l_xy = motion_model_on_point(local_dyaw_l, point.x, point.y, global_dyaw_l);
        std::pair<double, double> lidar_pt_r_xy = motion_model_on_point(local_dyaw_r, point.x, point.y, global_dyaw_r);
        Eigen::Vector4d lidar_pt_l(lidar_pt_l_xy.first, lidar_pt_l_xy.second, point.z, 1.0);
        Eigen::Vector4d lidar_pt_r(lidar_pt_r_xy.first, lidar_pt_r_xy.second, point.z, 1.0);

        // Eigen::Vector4d lidar_pt_l(point.x, point.y, point.z, 1.0);
        // Eigen::Vector4d lidar_pt_r(point.x, point.y, point.z, 1.0);

        // RCLCPP_INFO(logger, "Lidar Point Original: %f, %f, %f", point.x, point.y, point.z);
        // RCLCPP_INFO(logger, "Lidar Point Left: %f, %f, %f", lidar_pt_l(0), lidar_pt_l(1), lidar_pt_l(2));
        // RCLCPP_INFO(logger, "Lidar Point Right: %f, %f, %f", lidar_pt_r(0), lidar_pt_r(1), lidar_pt_r(2));

        double distance_l = std::sqrt(lidar_pt_l(0) * lidar_pt_l(0) + lidar_pt_l(1) * lidar_pt_l(1));
        double distance_r = std::sqrt(lidar_pt_r(0) * lidar_pt_r(0) + lidar_pt_r(1) * lidar_pt_r(1));

        // Apply projection matrix to LiDAR point
        Eigen::Vector3d transformed_l = projection_matrix_pair.first * lidar_pt_l;
        Eigen::Vector3d transformed_r = projection_matrix_pair.second * lidar_pt_r;

        // Divide by z coordinate for Euclidean normalization
        Eigen::Vector3d pixel_l(transformed_l(0) / transformed_l(2), transformed_l(1) / transformed_l(2), distance_l);
        Eigen::Vector3d pixel_r(transformed_r(0) / transformed_r(2), transformed_r(1) / transformed_r(2), distance_r);

        return std::make_pair(pixel_l, pixel_r);
    }

    std::pair<double, double> StateManager::motion_model_on_point(
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

    std::pair<double, double> StateManager::local_frame_to_local_frame(
        std::pair<double, double> local_frame_change,
        double dyaw)
    {
        double local_frame_dx = local_frame_change.first;
        double local_frame_dy = local_frame_change.second;

        double cmr_y = -local_frame_dx * std::sin(dyaw * M_PI / 180.0) + local_frame_dy * std::cos(dyaw * M_PI / 180.0);
        double cmr_x = local_frame_dx * std::cos(dyaw * M_PI / 180.0) + local_frame_dy * std::sin(dyaw * M_PI / 180.0);

        return std::make_pair(cmr_x, cmr_y);
    }

    State StateManager::get_prev_state(
        const rclcpp::Logger &logger,
        uint64_t frame_time
    ) {
        geometry_msgs::msg::TwistStamped::SharedPtr closest_vel_msg;
        geometry_msgs::msg::Vector3Stamped::SharedPtr closest_yaw_msg;
        geometry_msgs::msg::Vector3Stamped::SharedPtr yaw_msg;
        geometry_msgs::msg::TwistStamped::SharedPtr vel_msg;

        // Check if deque empty
        yaw_mutex.lock();
        if (yaw_deque.empty())
        {
            yaw_mutex.unlock();
            RCLCPP_WARN(logger, "Yaw deque is empty! No IMU data.");
            return std::make_pair(nullptr, nullptr);
        }

        yaw_msg = yaw_deque.back(); // in case all messages are before camera
        for (size_t i = 0; i < yaw_deque.size(); i++) {
            uint64_t yaw_time_ns = yaw_deque[i]->header.stamp.sec * 1e9 + yaw_deque[i]->header.stamp.nanosec;
            if (yaw_time_ns >= frame_time)
            {
                yaw_msg = yaw_deque[i];
                if (i > 0) { // interpolate with the previous
                    auto prev_yaw_msg = yaw_deque[i - 1];
                    auto prev_yaw_ts = prev_yaw_msg->header.stamp.sec * 1e9 + prev_yaw_msg->header.stamp.nanosec;
                    auto yaw = prev_yaw_msg->vector.z + (yaw_msg->vector.z - prev_yaw_msg->vector.z) * (frame_time - prev_yaw_ts) / (yaw_time_ns - prev_yaw_ts);
                    yaw_msg = std::make_shared<geometry_msgs::msg::Vector3Stamped>();
                    yaw_msg->vector.z = yaw;
                }
                break;
            }
        }
        yaw_mutex.unlock();

        // Check if deque empty
        vel_mutex.lock();
        if (vel_deque.empty())
        {
            vel_mutex.unlock();
            RCLCPP_WARN(logger, "Velocity deque is empty! No IMU data.");
            return std::make_pair(nullptr, nullptr);
        }

        vel_msg = vel_deque.back();
        // Iterate through deque to find the closest frame by timestamp
        for (const auto &vel : vel_deque)
        {
            uint64_t vel_time_ns = vel->header.stamp.sec * 1e9 + vel->header.stamp.nanosec;
            if (vel_time_ns >= frame_time)
            {
                vel_msg = vel;
                break;
            }
        }
        vel_mutex.unlock();

        return State(vel_msg, yaw_msg);
    }

    void StateManager::update_vel_deque(geometry_msgs::msg::TwistStamped::SharedPtr msg)
    {
        vel_mutex.lock();
        // Deque Management and Updating
        while (vel_deque.size() >= max_deque_size)
        {
            vel_deque.pop_front();
        }
        vel_deque.push_back(msg);
        vel_mutex.unlock();
    }

    void StateManager::update_yaw_deque(geometry_msgs::msg::Vector3Stamped::SharedPtr msg)
    {
        yaw_mutex.lock();
        // Deque Management and Updating
        while (yaw_deque.size() >= max_deque_size)
        {
            yaw_deque.pop_front();
        }
        yaw_deque.push_back(msg);
        yaw_mutex.unlock();
    }

} // namespace point_to_pixel