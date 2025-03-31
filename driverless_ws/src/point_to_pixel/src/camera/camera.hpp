#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <opencv2/opencv.hpp>
#include <videocapture.hpp>
#include <ocv_display.hpp>
#include <calibration.hpp>
#include "rclcpp/rclcpp.hpp"
#include <deque>

/**
 * @brief Finds the closest frame to a callback time from the image deque
 * 
 * @param img_deque Image deque with timestamps
 * @param callbackTime The time to find a matching frame for
 * @param logger ROS logger for error reporting
 * @return cv::Mat The closest frame
 */
cv::Mat find_closest_frame(
    const std::deque<std::pair<rclcpp::Time, cv::Mat>>& img_deque,
    const rclcpp::Time& callbackTime,
    const rclcpp::Logger& logger
);

/**
 * @brief Initialize ZED camera with calibration
 * 
 * @param cap The video capture object
 * @param device_id Device ID for the camera
 * @param map_left_x Output parameter for left x rectification map
 * @param map_left_y Output parameter for left y rectification map
 * @param map_right_x Output parameter for right x rectification map
 * @param map_right_y Output parameter for right y rectification map
 * @param logger ROS logger for status messages
 * @return bool Success status
 */
bool initialize_camera(
    sl_oc::video::VideoCapture& cap,
    int device_id,
    cv::Mat& map_left_x,
    cv::Mat& map_left_y,
    cv::Mat& map_right_x,
    cv::Mat& map_right_y,
    const rclcpp::Logger& logger
);

/**
 * @brief Captures and rectifies a frame from a ZED camera
 * 
 * @param cap The video capture object
 * @param map_left_x Left x rectification map
 * @param map_left_y Left y rectification map
 * @param map_right_x Right x rectification map
 * @param map_right_y Right y rectification map
 * @param use_inner_lens Whether to use inner or outer lens
 * @param logger ROS logger for error reporting
 * @return cv::Mat The rectified frame
 */
cv::Mat capture_and_rectify_frame(
    sl_oc::video::VideoCapture& cap,
    const cv::Mat& map_left_x,
    const cv::Mat& map_left_y,
    const cv::Mat& map_right_x,
    const cv::Mat& map_right_y,
    bool use_inner_lens,
    const rclcpp::Logger& logger
);

#endif