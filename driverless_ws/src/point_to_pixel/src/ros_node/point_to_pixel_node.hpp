#pragma once

// ROS2 imports
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
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
#define verbose 0  // Prints transform matrix and transformed pixel of every point
#define use_yolo 1 // 0: HSV Coloring | 1: YOLO Coloring
#define timing 1  // Prints timing suite at end of every callback
#define inner 1    // Uses inner lens of ZEDS (if 0 uses the outer lens)
#define save_frames 0 // Writes every 5th frame to img_log folder

struct Cone {
    geometry_msgs::msg::Point point;
    double distance;
    Cone(const geometry_msgs::msg::Point& p) : point(p) {
        distance = std::sqrt(p.x * p.x + p.y * p.y);
    }
};

/**
 * @brief Observer position represents where the car was when the one was observed.
 * Cur car position is where the car currently is. When the car moves, we calculate 
 * the change in x and y and add the changes to the observer position.
 * 
 * When a cone is observed, this information will be added to a queue.with which
 * we can perform data association on. As we do this, we should incremet the 
 * lifespan of the car.  
 */
struct ObsConeInfo {
   // Motion model changes are applied to these values
   float cur_car_to_observer_position_x;
   float cur_car_to_observer_position_y;

   // The observed positions of the cones in CMR frame 
   float observer_position_to_cone_x;
   float observer_position_to_cone_y;

   int lifespan;
};



class PointToPixelNode : public rclcpp::Node
{
public:
    // Constructor declaration
    PointToPixelNode();
    static constexpr int max_deque_size = 100;
    static constexpr int max_timesteps_in_cone_history = 10;

    #if use_yolo
    static constexpr char yolo_model_path[] = "src/point_to_pixel/config/yolov5_model_params.onnx";
    // static constexpr char yolo_model_path[] = "src/point_to_pixel/config/best164.onnx";
    #endif

    #if save_frames
    static constexpr int frame_interval = 5;
    static constexpr char save_path[] = "src/point_to_pixel/img_log/";
    #endif

    // static constexpr int zed_one_sn; // Left side zed
    // static constexpr int zed_two_sn; // Right side zed

private:
    // Image Deque
    std::deque<std::pair<uint64_t, cv::Mat>> img_deque_l;
    std::deque<std::pair<uint64_t, cv::Mat>> img_deque_r;
    std::deque<geometry_msgs::msg::TwistStamped::SharedPtr> velocity_deque;
    std::deque<geometry_msgs::msg::Vector3Stamped::SharedPtr> yaw_deque;

    std::queue<ObsConeInfo> yellow_cone_history;
    std::queue<ObsConeInfo> blue_cone_history;


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
    void cone_callback(const interfaces::msg::PPMConeArray::SharedPtr msg);
    void velocity_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg);
    void yaw_callback(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);

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

    std::mutex l_img_mutex;
    std::mutex r_img_mutex;
    std::mutex velocity_mutex;
    std::mutex yaw_mutex;
    std::mutex net_mutex;

    // ROS2 Publisher and Subscribers
    rclcpp::Publisher<interfaces::msg::ConeArray>::SharedPtr cone_pub_;
    rclcpp::Subscription<interfaces::msg::PPMConeArray>::SharedPtr cone_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr yaw_sub_;

    // Helper functions with implementations in separate files
    std::tuple<uint64_t, cv::Mat, uint64_t, cv::Mat> get_camera_frame(rclcpp::Time callbackTime);
    int get_cone_class(std::pair<Eigen::Vector3d, Eigen::Vector3d> pixel_pair,
                      std::pair<cv::Mat, cv::Mat> frame_pair,
                      std::pair<std::vector<cv::Mat>, std::vector<cv::Mat>> detection_pair);

    // Ordering function and helpers
    Cone findClosestCone(const std::vector<Cone>& cones);
    double calculateAngle(const Cone& from, const Cone& to);
    std::vector<Cone> orderConesByPathDirection(const std::vector<Cone>& unordered_cones);

    // Cone state propogation
    std::pair<double, double> getMotionEstimate(double velocity, double angle, double dt);
    std::pair<geometry_msgs::msg::TwistStamped::SharedPtr, geometry_msgs::msg::Vector3Stamped::SharedPtr> get_velocity_yaw(uint64_t callbackTime);

    std::thread launch_camera_communication();

    // Cone history and data association
    void classify_through_data_association(geometry_msgs::msg::Vector3 lidar_point);
    void motion_model_on_cone_history(std::queue<ObsConeInfo>& cone_history, std::pair<double, double> long_lat_change);
    void add_lidar_point_to_cone_history(std::queue<ObsConeInfo>& cone_history, geometry_msgs::msg::Vector3 lidar_point);
    void maintain_cone_history_lifespans(std::queue<ObsConeInfo>& cone_history);

    #if use_yolo
    cv::dnn::Net net; // YOLO Model
    #endif

    #if save_frames
    uint64_t camera_callback_count;
    void save_frame(std::pair<uint64_t, cv::Mat> frame_l, std::pair<uint64_t, cv::Mat> frame_r);
    #endif
};