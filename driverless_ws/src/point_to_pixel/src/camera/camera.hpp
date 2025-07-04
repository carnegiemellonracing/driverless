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

namespace camera {
    // USB Vendor & Product IDs for ZED cameras
    static constexpr uint16_t ZED_VENDOR_ID = 0x2b03;
    static constexpr uint16_t ZED_PRODUCT_ID = 0xf582;   // Original ZED
    static constexpr uint16_t ZED2_PRODUCT_ID = 0xf780;  // ZED 2


    struct Camera {
        sl_oc::video::VideoCapture &cap;
        cv::Mat map_left_x;
        cv::Mat map_left_y;
        cv::Mat map_right_x;
        cv::Mat map_right_y;
        int device_id;
        
        Camera(
            sl_oc::video::VideoCapture &cap_in,
            const cv::Mat &map_left_x_in,
            const cv::Mat &map_left_y_in,
            const cv::Mat &map_right_x_in,
            const cv::Mat &map_right_y_in,
            int device_id_in) :
            cap(cap_in),
            map_left_x(map_left_x_in),
            map_left_y(map_left_y_in),
            map_right_x(map_right_x_in),
            map_right_y(map_right_y_in),
            device_id(device_id_in)
        {}
    };

    /**
     * @brief Finds the closest frame to a callback time from the image deque
     *
     * @param img_deque Image deque with timestamps
     * @param callbackTime The time to find a matching frame for
     * @param logger ROS logger for error reporting
     * @return std::pair<uint64_t, cv::Mat> The closest frame with timestamp
     */
    std::pair<uint64_t, cv::Mat> find_closest_frame(
        const std::deque<std::pair<uint64_t, cv::Mat>> &img_deque,
        const rclcpp::Time &callbackTime,
        const rclcpp::Logger &logger
    );

    /**
     * @brief Dynamically assigns camera ids and corrects initial id assigmment. Also initializes 
     * ZED camera with intrinsic rectification matrices.
     * 
     * @param cam Camera struct to initialize
     * @param logger ROS logger for status messages
     * @return bool Success status
     */
    bool initialize_camera(
        Camera &cam,
        const rclcpp::Logger &logger
    );

    /**
     * @brief Captures and rectifies a frame from a ZED camera
     *
     * @param logger ROS logger for status messages
     * @param cam Camera struct containing capture and rectification maps
     * @param left_camera If using left sided zed set to true
     * @param use_inner_lens If using inner lenses set to true
     * @return std::pair<uint64_t, cv::Mat> The timestamp and rectified frame
     */
    std::pair<uint64_t, cv::Mat> capture_and_rectify_frame(
        const rclcpp::Logger &logger,
        const Camera &cam,
        bool left_camera,
        bool use_inner_lens
    );

    /**
     * @brief Capture and saves freeze frames for calibration
     *
     * @param logger ROS logger for status messages
     * @param left_cam Left camera struct
     * @param right_cam Right camera struct
     * @param l_img_mutex Mutex for left image deque
     * @param r_img_mutex Mutex for right image deque
     * @param img_deque_l Deque for left image frames
     * @param img_deque_r Deque for right image frames
     * @param use_inner_lens If using inner lenses set to true
     */
    void capture_freezes(
        const rclcpp::Logger &logger,
        const Camera &left_cam,
        const Camera &right_cam,
        std::mutex &l_img_mutex,
        std::mutex &r_img_mutex,
        std::deque<std::pair<uint64_t, cv::Mat>> &img_deque_l,
        std::deque<std::pair<uint64_t, cv::Mat>> &img_deque_r,
        bool use_inner_lens
    );

}