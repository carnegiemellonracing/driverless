#pragma once

#include <videocapture.hpp>
#include <ocv_display.hpp>
#include <calibration.hpp>
#include "rclcpp/rclcpp.hpp"
#include <opencv2/opencv.hpp>

// Standard Imports
#include <deque>
#include <cmath>
#include <mutex>
#include <filesystem>

namespace point_to_pixel {
    typedef std::pair<uint64_t, cv::Mat> StampedFrame;

    class CameraManager {
        public:
            // Public constructor
            CameraManager(sl_oc::video::VideoCapture &cap_in,
                   cv::Mat map_left_x_in,
                   cv::Mat map_left_y_in,
                   cv::Mat map_right_x_in,
                   cv::Mat map_right_y_in,
                   int device_id_in,
                   std::string save_path_in,
                   const int &max_deque_size_in)
                : cap(cap_in),
                  map_left_x(map_left_x_in),
                  map_left_y(map_left_y_in),
                  map_right_x(map_right_x_in),
                  map_right_y(map_right_y_in),
                  device_id(device_id_in), 
                  save_path(save_path_in),
                  max_deque_size(max_deque_size_in) {}

            /**
             * @brief Finds the closest frame to a callback time from the image deque
             *
             * @param callbackTime The time to find a matching frame for
             * @param logger ROS logger for error reporting
             * @return camera_manager::StampedFrame The closest frame with timestamp
             */
            StampedFrame find_closest_frame(
                const rclcpp::Time &callbackTime,
                const rclcpp::Logger &logger);

            /**
             * @brief Dynamically assigns camera ids and corrects initial id assigmment. Also initializes
             * ZED camera with intrinsic rectification matrices.
             *
             * @param logger ROS logger for status messages
             * @return bool Success status
             */
            bool initialize_camera(
                const rclcpp::Logger &logger);

            /**
             * @brief Captures and rectifies a frame from a ZED camera
             *
             * @param logger ROS logger for status messages
             * @param is_left_camera If using left sided zed set to true
             * @param use_inner_lens If using inner lenses set to true
             * @return camera_manager::StampedFrame The timestamp and rectified frame
             */
            StampedFrame capture_and_rectify_frame(
                const rclcpp::Logger &logger,
                bool is_left_camera,
                bool use_inner_lens);

            
            void capture_freezes(
                const rclcpp::Logger &logger,
                bool is_left_camera,
                bool use_inner_lens);

            /**
             * @brief Updates the deque with a new frame
             *
             * @param new_frame The new frame to add to the deque
             */
            void update_deque(
                StampedFrame new_frame
            );

        private:
            // USB Vendor & Product IDs for ZED cameras
            static constexpr uint16_t ZED_VENDOR_ID = 0x2b03;
            static constexpr uint16_t ZED_PRODUCT_ID = 0xf582;   // Original ZED
            static constexpr uint16_t ZED2_PRODUCT_ID = 0xf780;  // ZED 2

            sl_oc::video::VideoCapture &cap;
            cv::Mat map_left_x;
            cv::Mat map_left_y;
            cv::Mat map_right_x;
            cv::Mat map_right_y;
            int device_id;
            std::string save_path;

            std::deque<StampedFrame> img_deque;
            std::mutex img_mutex;
            int max_deque_size;
    };  
} // namespace point_to_pixel