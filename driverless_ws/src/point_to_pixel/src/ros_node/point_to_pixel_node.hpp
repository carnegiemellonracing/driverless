#pragma once

// ROS2 imports
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "interfaces/msg/ppm_cone_array.hpp"
#include "interfaces/msg/cone_list.hpp"
#include "interfaces/msg/cone_array.hpp"

// Project Headers
#include "../transform/transform.hpp"
#include "../camera/camera.hpp"
#include "../coloring/coloring.hpp"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::placeholders::_1;

// BUILD FLAGS
#define viz 1      // Prints color detection outputs of every point
#define verbose 1  // Prints transform matrix and transformed pixel of every point
#define use_yolo 0 // 0: HSV Coloring | 1: YOLO Coloring
#define timing 0   // Prints timing suite at end of every callback
#define inner 1    // Uses inner lens of ZEDS (if 0 uses the outer lens)

struct Cone {
    geometry_msgs::msg::Point point;
    double distance;
    Cone(const geometry_msgs::msg::Point& p) : point(p) {
        distance = std::sqrt(p.x * p.x + p.y * p.y);
    }
};

class PointToPixelNode : public rclcpp::Node
{
public:
    // Constructor declaration
    PointToPixelNode();
    static constexpr int max_deque_size = 10;
    // static constexpr int zed_one_sn; // Left side zed
    // static constexpr int zed_two_sn; // Right side zed

private:
    // Image Deque
    std::deque<std::pair<uint64_t, cv::Mat>> img_deque_l;
    std::deque<std::pair<uint64_t, cv::Mat>> img_deque_r;

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
    int get_cone_class(std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
                      std::pair<cv::Mat, cv::Mat> frame_pair,
                      std::pair<cv::Mat, cv::Mat> detection_pair);
    Cone findClosestCone(const std::vector<Cone>& cones);
    double calculateAngle(const Cone& from, const Cone& to);
    std::vector<Cone> orderConesByPathDirection(const std::vector<Cone>& unordered_cones);

#if use_yolo
    cv::dnn::Net net; // YOLO Model
#endif
};