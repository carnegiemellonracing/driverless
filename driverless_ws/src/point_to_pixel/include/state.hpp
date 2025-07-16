#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include <deque>

namespace point_to_pixel{
    typedef std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> state;

    class state_manager{
        public:
            // public contructor
            state_manager(const int &max_deque_size_in): max_deque_size(max_deque_size_in) {}

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
                int64_t left_frame_time,
                int64_t right_frame_time,
                int64_t lidar_frame_time,
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
             * @param frame_time Time of the frame in nanoseconds
             * @return std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr>
             */
            state get_prev_state(
                const rclcpp::Logger &logger,
                uint64_t frame_time
            );

            /**
             * @brief Update velocity deque
             * 
             * @param vel_msg Velocity message
             * @return void
             */
            void update_vel_deque(geometry_msgs::msg::TwistStamped::SharedPtr vel_msg);

            /**
             * @brief Update yaw deque
             * 
             * @param yaw_msg Yaw message
             * @return void
             */
            void update_yaw_deque(geometry_msgs::msg::Vector3Stamped::SharedPtr yaw_msg);
            

        private:
            int max_deque_size;
            std::mutex yaw_mutex;
            std::mutex vel_mutex;

            std::deque<geometry_msgs::msg::TwistStamped::SharedPtr> vel_deque;
            std::deque<geometry_msgs::msg::Vector3Stamped::SharedPtr> yaw_deque;

            /**
             * @brief Apply motion model on point
             * 
             * @param dv_frame_change Change in velocity in frame
             * @param point_x Point x
             * @param point_y Point y
             * @param dyaw Change in yaw
             * @return std::pair<double, double>
             */
            std::pair<double, double> motion_model_on_point(
                std::pair<double, double> dv_frame_change,
                double point_x,
                double point_y,
                double dyaw);

            /**
             * @brief Applies changes in local frame to local frame
             * 
             * @param local_frame_change Change in local frame
             * @param dyaw Change in yaw
             * @return std::pair<double, double>
             */
            std::pair<double, double> local_frame_to_local_frame(
                std::pair<double, double> local_frame_change,
                double dyaw);
    };
} // namespace point_to_pixel