#pragma once

// Project Headers
#include "../transform/transform.hpp"
#include "../camera/camera.hpp"
#include "../cones/cones.hpp"

// ROS2 imports
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "interfaces/msg/ppm_cone_array.hpp"
#include "interfaces/msg/cone_list.hpp"
#include "interfaces/msg/cone_array.hpp"

// Standard Imports
#include <deque>
#include <queue>
#include <memory>
#include <chrono>
#include <filesystem>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::placeholders::_1;

// BUILD FLAGS
#define verbose 0  // Prints transform matrix and transformed pixel of every point
#define use_yolo 1 // 0: HSV Coloring | 1: YOLO Coloring
#define timing 1  // Prints timing suite at end of every callback
#define inner 1    // Uses inner lens of ZEDS (if 0 uses the outer lens)
#define save_frames 1 // Writes every 5th frame to img_log folder

class PointToPixelNode : public rclcpp::Node
{
public:
    // Constructor declaration
    PointToPixelNode();
    static constexpr int max_deque_size = 100;

    // YOLO constants
    #if use_yolo
    static constexpr char yolo_model_path[] = "src/point_to_pixel/config/yolov5_model_params.onnx";
    // static constexpr char yolo_model_path[] = "src/point_to_pixel/config/best164.onnx";
    #endif

    // Frame saving constants
    #if save_frames
    static constexpr int frame_interval = 10;
    static constexpr char save_path[] = "src/point_to_pixel/img_log/";
    #endif

    // static constexpr int zed_one_sn; // Left side zed
    // static constexpr int zed_two_sn; // Right side zed

private:
    // ROS2 Publisher and Subscribers
    rclcpp::Publisher<interfaces::msg::ConeArray>::SharedPtr cone_pub_;
    rclcpp::Subscription<interfaces::msg::PPMConeArray>::SharedPtr cone_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr yaw_sub_;

    // Data Structure Declarations
    std::deque<std::pair<uint64_t, cv::Mat>> img_deque_l;
    std::deque<std::pair<uint64_t, cv::Mat>> img_deque_r;
    std::deque<geometry_msgs::msg::TwistStamped::SharedPtr> velocity_deque;
    std::deque<geometry_msgs::msg::Vector3Stamped::SharedPtr> yaw_deque;
    #if save_frames
    std::queue<std::tuple<uint64_t, cv::Mat, uint64_t, cv::Mat>> save_queue;
    #endif

    // Mutexes Declarations for thread safety
    std::mutex l_img_mutex;
    std::mutex r_img_mutex;
    std::mutex velocity_mutex;
    std::mutex yaw_mutex;
    #if save_frames
    std::mutex save_mutex;
    #endif
    
    // ROS Arg Parameters
    std::vector<double> param_l;
    std::vector<double> param_r;
    Eigen::Matrix<double, 3, 4> projection_matrix_l;
    Eigen::Matrix<double, 3, 4> projection_matrix_r;
    double confidence_threshold;
    cv::Scalar yellow_filter_high;
    cv::Scalar yellow_filter_low;
    cv::Scalar blue_filter_high;
    cv::Scalar blue_filter_low;
    cv::Scalar orange_filter_high;
    cv::Scalar orange_filter_low;

    // Camera params
    sl_oc::video::VideoParams params;
    
    // Camera objects
    sl_oc::video::VideoCapture cap_l;
    sl_oc::video::VideoCapture cap_r;
    camera::Camera left_cam;
    camera::Camera right_cam;
    
    #if save_frames
    uint64_t camera_callback_count;
    #endif

    #if use_yolo
    cv::dnn::Net net; // YOLO Model
    #endif

    // Threads for camera callback and frame saving
    std::thread launch_camera_communication();
    #if save_frames
    std::thread launch_frame_saving();
    #endif

    /**
     * @brief Triggers full point to pixel pipeline. First retrieves closest camera frames L and R to lidar timestamp and
     * runs YOLO detection on both frames. Then loops through each cone point in topic callback, transforms it to camera space
     * and determines the color of the cone by seeing which bounding box it falls under. Final cone vectors are then ordered 
     * and published.
     * 
     * @param msg from /cpp_cones topic of type PPMConeArray
     * @return void but publishes to /perc_cones topic of type ConeArray
     */
    void cone_callback(const interfaces::msg::PPMConeArray::SharedPtr msg);

    /**
     * @brief Topic callback for velocity. Updates the velocity deque with the latest velocity message.
     * 
     * @param msg from /filter/twist topic of type TwistStamped
     * @return void
     */
    void velocity_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);

    /**
     * @brief Topic callback for yaw. Updates the yaw deque with the latest yaw message.
     * 
     * @param msg from /filter/euler topic of type Vector3Stamped
     * @return void
     */
    void yaw_callback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);

    /**
     * @brief Threaded callback for capturing frames from left and right cameras. Captures and rectifies frames 
     * from both cameras, then updates the image deques with the captured frame and corresponding timestamp.
     */
    void camera_callback();
    
    /**
     * @brief Captures freeze frames for calibration
     */
    void capture_freezes();

    /**
     * @brief Wrapper function responsible for retrieving camera frames closest to the lidar timestamp.
     * 
     * @param callbackTime Time of the lidar frame
     * @return std::tuple<uint64_t, cv::Mat, uint64_t, cv::Mat> Tuple containing the timestamps and frames from both cameras
     */
    std::tuple<uint64_t, cv::Mat, uint64_t, cv::Mat> get_camera_frame(rclcpp::Time callbackTime);

    /**
     * @brief Wrapper function for retrieving the color of cone by combining output from both cameras
     * 
     * @param pixel_pair Pixel coordinates in both cameras
     * @param frame_pair Frames from both cameras
     * @param detection_pair YOLO detection results (Unused if not using YOLO)
     */
    int get_cone_class(std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
                      std::pair<cv::Mat, cv::Mat> frame_pair,
                      std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> detection_pair);
    
    #if save_frames
    /**
     * @brief Saves the frame to the specified path.
     * 
     * @param logger ROS Logger for logging messages
     * @param frame_tuple Tuple containing the timestamps and frames from both cameras
     */
    void save_frame(const rclcpp::Logger &logger, std::tuple<uint64_t, cv::Mat, uint64_t, cv::Mat> frame_tuple);
    #endif
};