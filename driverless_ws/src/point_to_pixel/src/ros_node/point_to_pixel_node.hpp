#pragma once

// ROS2 imports
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "interfaces/msg/ppm_cone_array.hpp"
#include "interfaces/msg/cone_list.hpp"
#include "interfaces/msg/cone_array.hpp"

// zed-open-capture library header
#include <videocapture.hpp>
#include <ocv_display.hpp>
#include <calibration.hpp>

// Standard Imports
#include <deque>
#include <memory>
#include <chrono>

// Project Headers
#include "../transform/transform.hpp"
#include "../camera/camera.hpp"
#include "../coloring/coloring.hpp"

// Flags
#define VIZ 0      // Prints color detection outputs of every point
#define VERBOSE 0  // Prints transform matrix and transformed pixel of every point
#define USE_YOLO 0 // 0: HSV Coloring | 1: YOLO Coloring
#define TIMING 0   // Prints timing suite at end of every callback
#define INNER 1    // Uses inner lens of ZEDS (if 0 uses the outer lens)

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::placeholders::_1;

static constexpr int max_deque_size = 10;

class PointToPixelNode : public rclcpp::Node
{
public:
    // Constructor declaration
    PointToPixelNode();

private:
    // Image Deque
    std::deque<std::pair<rclcpp::Time, cv::Mat>> img_deque_l;
    std::deque<std::pair<rclcpp::Time, cv::Mat>> img_deque_r;

    // ROS2 Parameters
    Eigen::Matrix<double, 3, 4> projection_matrix_l;
    Eigen::Matrix<double, 3, 4> projection_matrix_r;

    double confidence_threshold;
    cv::Scalar yellow_filter_high;
    cv::Scalar yellow_filter_low;
    cv::Scalar blue_filter_high;
    cv::Scalar blue_filter_low;
    cv::Scalar orange_filter_high;
    cv::Scalar orange_filter_low;

    // Topic Callback Functions
    void topic_callback(const interfaces::msg::PPMConeArray::SharedPtr msg);

    // Camera Callback Functions
    void camera_callback();
    rclcpp::TimerBase::SharedPtr camera_timer_;

    // Camera Objects and Parameters
    sl_oc::video::VideoParams params;
    sl_oc::video::VideoCapture cap_l;
    sl_oc::video::VideoCapture cap_r;

    sl_oc::video::Frame canvas;
    sl_oc::video::Frame frame_l;
    sl_oc::video::Frame frame_r;

    // Rectification maps
    cv::Mat map_left_x_ll, map_left_y_ll;
    cv::Mat map_right_x_lr, map_right_y_lr;

    cv::Mat map_left_x_rl, map_left_y_rl;
    cv::Mat map_right_x_rr, map_right_y_rr;

    // ROS2 Publisher and Subscribers
    rclcpp::Publisher<interfaces::msg::ConeArray>::SharedPtr publisher_;
    rclcpp::Subscription<interfaces::msg::PPMConeArray>::SharedPtr subscriber_;

    // Helper functions with implementations in separate files
    std::pair<cv::Mat, cv::Mat> get_camera_frame(rclcpp::Time callbackTime);
    int get_cone_class(std::pair<Eigen::Vector2d, Eigen::Vector2d> pixel_pair,
                      std::pair<cv::Mat, cv::Mat> frame_pair,
                      std::pair<cv::Mat, cv::Mat> detection_pair);

#if USE_YOLO
    cv::dnn::Net net; // YOLO Model
#endif
};