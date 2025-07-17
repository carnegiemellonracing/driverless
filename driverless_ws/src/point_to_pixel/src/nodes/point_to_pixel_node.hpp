#pragma once

// Headers
#include "managers/state_manager.hpp"
#include "managers/camera_manager.hpp"
#include "cones/cones.hpp"
#include "cones/predictors/svm.hpp"
#include "cones/predictors/general_predictor.hpp"
#include "cones/predictors/yolo_predictor.hpp"
#include "cones/predictors/hsv_predictor.hpp"

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
#define verbose 0     // Prints transform matrix and transformed pixel of every point
#define use_yolo 1    // 0: HSV Coloring | 1: YOLO Coloring
#define timing 1      // Prints timing suite at end of every callback
#define inner 1       // Uses inner lens of ZEDS (if 0 uses the outer lens)
#define save_frames 0 // Writes every 5th frame to img_log folder

namespace point_to_pixel
{
    class PointToPixelNode : public rclcpp::Node
    {
    public:
        // Constructor declaration
        PointToPixelNode();

    private:
        // Constants
        static constexpr int max_deque_size = 100;
        static constexpr double svm_C = 5.0;

#if use_yolo
        static constexpr char yolo_model_path[] = "src/point_to_pixel/data/yolov5_model_params.onnx";
#endif // use_yolo

        static constexpr char save_path[] = "src/point_to_pixel/freezes/";
#if save_frames
        static constexpr int frame_interval = 10;
#endif // save_frames

        // ROS2 Publisher and Subscribers
        rclcpp::Publisher<interfaces::msg::ConeArray>::SharedPtr cone_pub_;
        rclcpp::Subscription<interfaces::msg::PPMConeArray>::SharedPtr cone_sub_;
        rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_sub_;
        rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr yaw_sub_;

        // Predictor
        std::unique_ptr<GeneralPredictor> predictor_;

// Data Structure Declarations
#if save_frames
        std::queue<std::tuple<uint64_t, cv::Mat, uint64_t, cv::Mat>> save_queue;
#endif // save_frames

// Mutexes
#if save_frames
        std::mutex save_mutex;
#endif // save_frames

        // Default Params
        std::vector<double> default_matrix{12.0f, 0.0f};
        float confidence_threshold_default = 0.25f;
        std::vector<long int> ly_filter_default{0, 0, 0};
        std::vector<long int> uy_filter_default{0, 0, 0};
        std::vector<long int> lb_filter_default{0, 0, 0};
        std::vector<long int> ub_filter_default{255, 255, 255};
        std::vector<long int> lo_filter_default{255, 255, 255};
        std::vector<long int> uo_filter_default{255, 255, 255};

        // ROS Arg Parameters
        std::vector<double> param_l;
        std::vector<double> param_r;
        Eigen::Matrix<double, 3, 4> projection_matrix_l;
        Eigen::Matrix<double, 3, 4> projection_matrix_r;
        float confidence_threshold;
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
        CameraManager *left_cam_;
        CameraManager *right_cam_;

        // State objects
        StateManager *state_manager_;

#if save_frames
        uint64_t camera_callback_count;
#endif // save_frames

        // Threads for camera callback and frame saving
        std::thread launch_camera_communication();
#if save_frames
        std::thread launch_frame_saving();
#endif // save_frames

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
         * @brief Threaded callback for capturing frames from left and right cameras. Captures and rectifies frames
         * from both cameras, then updates the image deques with the captured frame and corresponding timestamp.
         */
        void camera_callback();

        /**
         * @brief Helper function to convert ConeClass enum to integer for legacy compatibility
         *
         * @param cone_class ConeClass enum value
         * @return int representation (0=orange, 1=yellow, 2=blue, -1=unknown)
         */
        int cone_class_to_int(ConeClass cone_class);

#if save_frames
        /**
         * @brief Saves the frame to the specified path.
         *
         * @param logger ROS Logger for logging messages
         * @param frame_tuple Tuple containing the timestamps and frames from both cameras
         */
        void save_frame(const rclcpp::Logger &logger, std::tuple<uint64_t, cv::Mat, uint64_t, cv::Mat> frame_tuple);
#endif // save_frames
    };
} // namespace point_to_pixel