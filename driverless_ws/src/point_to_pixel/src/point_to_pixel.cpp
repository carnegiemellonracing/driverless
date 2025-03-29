// Open CV and Eigen
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/dnn.hpp> // Darknet Neural Network Module

// zed-open-capture library header
#include <videocapture.hpp>
#include <ocv_display.hpp>
#include <calibration.hpp>

// Standard Imports
#include <cstdio>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <filesystem>
#include <cmath>
#include <unordered_set>
#include <algorithm>

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// ROS2 imports
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "interfaces/msg/ppm_cone_array.hpp"
#include "interfaces/msg/cone_list.hpp"
#include "interfaces/msg/cone_array.hpp"

#include "geometry_msgs/msg/TwistStamped.hpp"
#include "geometry_msgs/msg/QuaternionStamped.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;

// ---------------------------------------------------------------------------
//    FLAGS
// ---------------------------------------------------------------------------
#define VIZ 0 // Prints color detection outputs of every point
#define VERBOSE 0 // Prints transform matrix and transformed pixel of every point
#define YOLO 0 // 0: HSV Coloring | 1: YOLO Coloring
#define TIMING 1 // Prints timing suite at end of every callback

struct Cone {
    geometry_msgs::msg::Point point;
    double distance;
    Cone(const geometry_msgs::msg::Point& p) : point(p) {
        distance = std::sqrt(p.x * p.x + p.y * p.y);
    }
};

class Point_To_Pixel_Node : public rclcpp::Node
{
  public:
    // Constructor declaration
    Point_To_Pixel_Node(); 

  private:
    // Image Deque
    std::deque<std::pair<rclcpp::Time, cv::Mat>> img_deque;

    // ROS2 Parameters
    Eigen::Matrix<double, 3, 4> projection_matrix;
    double CONFIDENCE_THRESHOLD;
    cv::Scalar yellow_filter_high;
    cv::Scalar yellow_filter_low;
    cv::Scalar blue_filter_high;
    cv::Scalar blue_filter_low;
    cv::Scalar orange_filter_high;
    cv::Scalar orange_filter_low;


    // Topic Callback Functions
    void topic_callback(const interfaces::msg::PPMConeArray::SharedPtr msg); 
    Eigen::Vector2d transform(geometry_msgs::msg::Vector3 &point);

    // Camera Callback Functions
    void camera_callback();
    rclcpp::TimerBase::SharedPtr camera_timer_;
    cv::Mat getCameraFrame(rclcpp::Time callbackTime);
    

    // Coloring Functions
    #if YOLO
          cv::dnn::Net net; // YOLO Model
      int get_yolo_color(Eigen::Vector2d& pixel, cv::Mat image, int cols, int rows);
    #else
      int get_hsv_color(Eigen::Vector2d& pixel, cv::Mat image);
    #endif

    // Camera Objects and Parameters
    sl_oc::video::VideoParams params;
    sl_oc::video::VideoCapture cap_0;
    sl_oc::video::VideoCapture cap_1;

    sl_oc::video::Frame canvas;
    sl_oc::video::Frame frame_0;
    sl_oc::video::Frame frame_1;

    cv::Mat map_left_x, map_left_y;
    cv::Mat map_right_x, map_right_y;

    // ROS2 Publisher and Subscribers
    rclcpp::Publisher<interfaces::msg::ConeArray>::SharedPtr publisher_;
    rclcpp::Subscription<interfaces::msg::PPMConeArray>::SharedPtr subscriber_;

    // rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr twist_subscriber_;
    // rclcpp::Subscription<geometry_msgs::msg::QuaternionStamped>::SharedPtr quat_subscriber_;

    Cone findClosestCone(const std::vector<Cone>& cones);
    double calculateAngle(const Cone& from, const Cone& to);
    std::vector<Cone> orderConesByPathDirection(const std::vector<Cone>& unordered_cones);
};

// TODO: FIX ZED static ID per this forum https://github.com/stereolabs/zed-ros-wrapper/issues/94

// Constructor definition
Point_To_Pixel_Node::Point_To_Pixel_Node() : Node("point_to_pixel"),
                                              params([]() {sl_oc::video::VideoParams p; p.res = sl_oc::video::RESOLUTION::HD1080; p.fps = sl_oc::video::FPS::FPS_30; return p;}()),
                                              cap_0(sl_oc::video::VideoCapture(params)),
                                              cap_1(sl_oc::video::VideoCapture(params))
{
  // ---------------------------------------------------------------------------
  //                                 CAMERAS
  // ---------------------------------------------------------------------------

  // Second Camera Code
  // // Checks if the video capture object was properly declared and initialized
  // if(!(this->cap_0).initializeVideo(0))
  // {
  //   RCLCPP_ERROR(this->get_logger(), "Cannot open camera 1 video capture");
  //   rclcpp::shutdown(); // Shutdown node
  // }

  // RCLCPP_INFO(this->get_logger(), "Connected to ZED camera 0. %s", (this->cap_0).getDeviceName().c_str());

  // Checks if the video capture object was properly declared and initialized

  if(!(this->cap_1).initializeVideo(0))
  {
    RCLCPP_ERROR(this->get_logger(), "Cannot open camera 1 video capture");
    rclcpp::shutdown(); // Shutdown node
  }

  // Rectify Image
  int sn = this->cap_1.getSerialNumber();

  // ----> Retrieve calibration file from Stereolabs server
  std::string calibration_file;
  // ZED Calibration
  unsigned int serial_number = sn;
  // Download camera calibration file
  if (!sl_oc::tools::downloadCalibrationFile(serial_number, calibration_file))
  {
    std::cerr << "Could not load calibration file from Stereolabs servers" << std::endl;
  }

  // Get Frame size
  int w, h;
  this->cap_1.getFrameSize(w, h);

  // Initialize calibration
  cv::Mat cameraMatrix_left, cameraMatrix_right;
  sl_oc::tools::initCalibration(calibration_file, cv::Size(w / 2, h), 
                                this->map_left_x, this->map_left_y, this->map_right_x, this->map_right_y,
                                cameraMatrix_left, cameraMatrix_right);
  // Set auto exposure and brightness
  this->cap_1.setAECAGC(true);

  RCLCPP_INFO(this->get_logger(), "ZED Camera 1 Ready. %s", (this->cap_1).getDeviceName().c_str());

  // ---------------------------------------------------------------------------
  //                               PARAMETERS
  // ---------------------------------------------------------------------------

  // Initialize Empty IMG Deque
  this->img_deque = {};

  // Projection matrix that takes LiDAR points to pixels
  std::vector<double> param_default(12, 1.0f); // NEED TO CHANGE TO RETRIEVE WHEN BUILDING
  this->declare_parameter("projection_matrix", param_default);

  // Threshold that determines whether it reports the color on a cone or not
  this->declare_parameter("confidence_threshold", 0.05); 

  #if YOLO
    // YOLO Stuff
    this->net = cv::dnn::readNetFromONNX("src/point_to_pixel/config/best164.onnx");

    if (this->net.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Error Loading YOLO Model");
      rclcpp::shutdown();
    }

    RCLCPP_INFO(this->get_logger(), "YOLO Model %s");
    
    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  #endif

  std::vector<long int> ly_filter_default{0, 0, 0};
  std::vector<long int> uy_filter_default{0, 0, 0};
  std::vector<long int> lb_filter_default{0, 0, 0};
  std::vector<long int> ub_filter_default{255, 255, 255};
  std::vector<long int> lo_filter_default{255, 255, 255};
  std::vector<long int> uo_filter_default{255, 255, 255};

  // Color Parameters
  this->declare_parameter("yellow_filter_high", ly_filter_default);
  this->declare_parameter("yellow_filter_low", uy_filter_default);
  this->declare_parameter("blue_filter_high", lb_filter_default);
  this->declare_parameter("blue_filter_low", ub_filter_default);
  this->declare_parameter("orange_filter_high", lo_filter_default);
  this->declare_parameter("orange_filter_low", uo_filter_default);

  // Load Projection Matrix
  std::vector<double> param = this->get_parameter("projection_matrix").as_double_array();
  this->projection_matrix = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(param.data());

  // Load Confidence Threshold
  this->CONFIDENCE_THRESHOLD = this->get_parameter("confidence_threshold").as_double();

  // Load Color Filter Params
  std::vector<long int> uy_filt_arr = this->get_parameter("yellow_filter_high").as_integer_array();
  std::vector<long int> ly_filt_arr = this->get_parameter("yellow_filter_low").as_integer_array();
  std::vector<long int> lb_filt_arr = this->get_parameter("blue_filter_low").as_integer_array();
  std::vector<long int> ub_filt_arr = this->get_parameter("blue_filter_high").as_integer_array();
  std::vector<long int> lo_filt_arr = this->get_parameter("orange_filter_low").as_integer_array();
  std::vector<long int> uo_filt_arr = this->get_parameter("orange_filter_high").as_integer_array();

  this->yellow_filter_high = cv::Scalar(uy_filt_arr[0], uy_filt_arr[1], uy_filt_arr[2]);
  this->yellow_filter_low = cv::Scalar(ly_filt_arr[0], ly_filt_arr[1], ly_filt_arr[2]);
  this->blue_filter_high = cv::Scalar(ub_filt_arr[0], ub_filt_arr[1], ub_filt_arr[2]);
  this->blue_filter_low = cv::Scalar(lb_filt_arr[0], lb_filt_arr[1], lb_filt_arr[2]);
  this->orange_filter_high = cv::Scalar(uo_filt_arr[0], uo_filt_arr[1], uo_filt_arr[2]);
  this->orange_filter_low = cv::Scalar(lo_filt_arr[0], lo_filt_arr[1], lo_filt_arr[2]);

  std::chrono::seconds duration(2);
  rclcpp::sleep_for(duration);

  // this->twist = geometry_msgs::msg::TwistStamped();
  // this->quat = geometry_msgs::msg::QuaternionStamped();

  // ---------------------------------------------------------------------------
  //                              ROS2 OBJECTS
  // ---------------------------------------------------------------------------
  
  // Publisher that returns colored cones
  publisher_ = this->create_publisher<interfaces::msg::ConeArray>("/perc_cones", 10);
  
  // Subscriber that reads the input topic that contains an array of cone_point arrays from LiDAR stack
  subscriber_ = this->create_subscription<interfaces::msg::PPMConeArray>(
    "/cpp_cones", 
    10, 
    [this](const interfaces::msg::PPMConeArray::SharedPtr msg) {this->topic_callback(msg);}
  );

  // twist_subscriber_ = this->create_subscription<geometry_msgs::msg::TwistStamped>(
  //   "/filter/twist",
  //   10,
  //   [this](const geometry_msgs::msg::TwistStamped::SharedPtr msg) {this->twist_callback(msg);}
  // );

  // quat_subscriber_ = this->create_subscription<geometry_msgs::msg::QuaternionStamped>(
  //   "/filter/quat",
  //   10,
  //   [this](const geometry_msgs::msg::QuaternionStamped::SharedPtr msg) {this->quat_callback(msg);}
  // );

  // Camera Callback (10 fps)
  camera_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(40),
    [this](){this->camera_callback();}
  );

  // ---------------------------------------------------------------------------
  //                       INITIALIZATION COMPLETE SEQUENCE
  // ---------------------------------------------------------------------------
  // Retrieve freeze frame for calibration
  const sl_oc::video::Frame frame_1 = this->cap_1.getLastFrame();

  cv::Mat frameBGR_1, left_raw, left_rect, right_raw, right_rect;
  if (frame_1.data != nullptr){
      cv::Mat frameYUV_1 = cv::Mat(frame_1.height, frame_1.width, CV_8UC2, frame_1.data);

      cv::cvtColor(frameYUV_1,frameBGR_1,cv::COLOR_YUV2BGR_YUYV);

      // ----> Extract left and right images from side-by-side
      left_raw = frameBGR_1(cv::Rect(0, 0, frameBGR_1.cols / 2, frameBGR_1.rows));
      right_raw = frameBGR_1(cv::Rect(frameBGR_1.cols / 2, 0, frameBGR_1.cols / 2, frameBGR_1.rows));

      // ----> Apply rectification
      cv::remap(left_raw, left_rect, this->map_left_x, this->map_left_y, cv::INTER_LINEAR );
      cv::remap(right_raw, right_rect, this->map_right_x, this->map_right_y, cv::INTER_LINEAR );
      frameBGR_1(cv::Rect(0, 0, frameBGR_1.cols / 2, frameBGR_1.rows)) = left_rect;
      frameBGR_1(cv::Rect(frameBGR_1.cols / 2, 0, frameBGR_1.cols / 2, frameBGR_1.rows)) = right_rect;
      
  }
  // Save freeze.png
  cv::imwrite("/home/chip/Documents/driverless/driverless_ws/src/point_to_pixel/config/freeze.png", left_rect);


  // Initialization Complete Message Suite
  RCLCPP_INFO(this->get_logger(), "Point to Pixel Node INITIALIZED");
  #if VERBOSE
    RCLCPP_INFO(this->get_logger(), "Verbose Logging On");
  #endif
  #if YOLO
      RCLCPP_INFO(this->get_logger(), "Using YOLO Color Detection");
  #else
      RCLCPP_INFO(this->get_logger(), "Using HSV Color Detection");
  #endif
};

// Maps 3D point to 2D Pixel
Eigen::Vector2d Point_To_Pixel_Node::transform(geometry_msgs::msg::Vector3& point)
{

  #if VERBOSE
  // Create a stringstream to log the matrix
  std::stringstream ss;

    // Iterate over the rows and columns of the matrix and format the output
    for (int i = 0; i < this->projection_matrix.rows(); ++i){
      for (int j = 0; j < this->projection_matrix.cols(); ++j){
        ss << this->projection_matrix(i, j) << " ";
      }
      ss << "\n";
    }
    // Log the projection_matrix using ROS 2 logger
    RCLCPP_INFO(this->get_logger(), "3x4 projection_matrix:\n%s", ss.str().c_str());
  #endif

  // Convert point from topic type (geometry_msgs/msg/Vector3) to Eigen Vector3d
  Eigen::Vector4d lidar_pt(point.x, point.y, point.z, 1.0);

  // Apply projection matrix to LiDAR point
  Eigen::Vector3d transformed = this->projection_matrix * lidar_pt;

  // Divide by z coordinate for Euclidean normalization
  Eigen::Vector2d pixel_1 (transformed(0)/transformed(2), transformed(1)/transformed(2));

  return pixel_1;
}


// Returns closest frame to callback time
cv::Mat Point_To_Pixel_Node::getCameraFrame(rclcpp::Time callbackTime)
{
  // Initialize Variables
  int64 bestDiff = INT64_MAX;
  cv::Mat closestFrame;
  
  // Check if deque empty
  if (this->img_deque.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Image deque is empty! Cannot find matching frame.");
    return cv::Mat(); // Return empty matrix
  }

  // Iterate through deque, simple best diff calculation
  for (const auto& frame: this->img_deque) {
    int64 timeDiff = abs(frame.first.nanoseconds() - callbackTime.nanoseconds());

    if (timeDiff < bestDiff) {
      closestFrame = frame.second;
      bestDiff = timeDiff;
    }
  }
  
  return closestFrame;
}

#if YOLO
  // Checks if pixel in bounding box, returns bounding box with highest score
  int Point_To_Pixel_Node::get_yolo_color(Eigen::Vector2d& pixel, cv::Mat detections, int cols, int rows)
  {
    int x = static_cast<int>(pixel(0));
    int y = static_cast<int>(pixel(1));

    // Post Processing
    const float x_factor = cols / 640.0;
    const float y_factor = rows / 640.0;

    // Loop through all detection
    for (int i = 0; i < detections.rows; ++i) 
    {
      float confidence = detections.at<float>(i, 4);

      if (confidence >= this->CONFIDENCE_THRESHOLD)
      {
        // Get bounding box coordinates
        float cx = detections.at<float>(i, 0) * x_factor;
        float cy = detections.at<float>(i, 1) * y_factor;
        float width = detections.at<float>(i, 2) * x_factor;
        float height = detections.at<float>(i, 3) * y_factor;
        
        // Calculate the bounding box corners
        float left = cx - width/2;
        float top = cy - height/2;
        float right = cx + width/2;
        float bottom = cy + height/2;
        
        // If pixel is inside the bounding box
        if (left <= x && x <= right && top <= y && y <= bottom)
        {
          // Find the highest class score
          float max_class_score = 0;
          int class_id = -1;
          
          // Assuming class scores start after index 5 (++j)
          for (int j = 5; j < detections.cols; ++j) 
          {
            float class_score = detections.at<float>(i, j);
            if (class_score > max_class_score) 
            {
              max_class_score = class_score;
              class_id = j - 5;  // Adjust index to get the actual class ID
            }
          }
          
          // Map class_id to cone color IDK WHAT THE CLASS_ID'S ARE
          int cone_color = -1;
          if (class_id >= 0) {
            switch(class_id) {
              case 0: cone_color = 0; break;  // Orange cone
              case 1: cone_color = 1; break;  // Yellow cone
              case 2: cone_color = 2; break;  // Blue cone
              default: cone_color = -1; break; // Unknown
            }
            return std::make_pair(cone_color, confidence);
          }
        }
      }
    }
    
    // No detection 
    return std::make_pair(-1, 0.0);
  }
#endif

// Applies HSV Filter to image frame
// Returns max of all colored pixels within ROI 
int Point_To_Pixel_Node::get_hsv_color(Eigen::Vector2d& pixel, cv::Mat img)
{
  // Ratio of color in relation to all other colors
  const double RATIO_THRESHOLD = 1.5;

  // Setup region of interest
  int side_length = 25;
  int x = static_cast<int>(pixel(0));
  int y = static_cast<int>(pixel(1));
  int height = img.rows;
  int width = img.cols;
  int x_min = std::max(0, x - side_length);
  int x_max = std::min(width, x + side_length);
  int y_min = std::max(0, y - side_length);
  int y_max = std::min(height, y + side_length);

  #if VIZ
    // Checks if transformed point is out of Frame
    RCLCPP_INFO(this->get_logger(), "point out of frame? (y_min >= y_max): %d, %d, %s", y_min, y_max, y_min >= y_max ? "true": "false");
    RCLCPP_INFO(this->get_logger(), "point out of frame? (x_min >= x_max): %d, %d, %s", x_min, x_max, x_min >= x_max ? "true": "false");
  #endif

  // Transformed point out of frame
  if (x_min >= x_max || y_min >= y_max) {
      return -1;
  }

  // Extract ROI and convert to HSV
  cv::Mat roi = img(cv::Range(y_min, y_max), cv::Range(x_min, x_max));
  cv::Mat hsv_roi;
  cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);

  // Define HSV color ranges
  std::vector<std::pair<cv::Scalar, cv::Scalar>> yellow_ranges = {
    {this->yellow_filter_low, this->yellow_filter_high}
  };

  std::vector<std::pair<cv::Scalar, cv::Scalar>> blue_ranges = {
    {this->blue_filter_low, this->blue_filter_high}
  };

  std::vector<std::pair<cv::Scalar, cv::Scalar>> orange_ranges = {
    {this->orange_filter_low, this->orange_filter_high}
  };


  // Create color masks
  cv::Mat yellow_mask = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
  cv::Mat blue_mask = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
  cv::Mat orange_mask = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
  cv::Mat temp_mask;

  // Apply color masks
  for (const auto& range : yellow_ranges) {
      cv::inRange(hsv_roi, range.first, range.second, temp_mask);
      cv::bitwise_or(yellow_mask, temp_mask, yellow_mask);
  }
  for (const auto& range : blue_ranges) {
      cv::inRange(hsv_roi, range.first, range.second, temp_mask);
      cv::bitwise_or(blue_mask, temp_mask, blue_mask);
  }
  for (const auto& range : orange_ranges) {
      cv::inRange(hsv_roi, range.first, range.second, temp_mask);
      cv::bitwise_or(orange_mask, temp_mask, orange_mask);
  }

  // Calculate color percentages
  double total_pixels = (y_max - y_min) * (x_max - x_min);
  double yellow_pixels = cv::countNonZero(yellow_mask);
  double blue_pixels = cv::countNonZero(blue_mask);
  double orange_pixels = cv::countNonZero(orange_mask);
  double yellow_percentage = yellow_pixels / total_pixels;
  double blue_percentage = blue_pixels / total_pixels;
  double orange_percentage = orange_pixels / total_pixels;

  // Print out the color percentages
  #if VIZ
    std::cout << "Yellow Percentage: " << yellow_percentage * 100 << "%" << std::endl;
    std::cout << "Blue Percentage: " << blue_percentage * 100 << "%" << std::endl;
    std::cout << "Orange Percentage: " << orange_percentage * 100 << "%" << std::endl;
  #endif

  // Determine cone color
  if (orange_percentage > this->CONFIDENCE_THRESHOLD && orange_percentage > std::max(yellow_percentage, blue_percentage) * RATIO_THRESHOLD) {
      return 0;
  } else if (yellow_percentage > this->CONFIDENCE_THRESHOLD && yellow_percentage > std::max(blue_percentage, orange_percentage) * RATIO_THRESHOLD) {
      return 1;
  } else if (blue_percentage > this->CONFIDENCE_THRESHOLD && blue_percentage > std::max(yellow_percentage, orange_percentage) * RATIO_THRESHOLD) {
      return 2;
  }
  return -1;
}

Cone Point_To_Pixel_Node::findClosestCone(const std::vector<Cone>& cones) {
    if (cones.empty()) {
        throw std::runtime_error("Empty cone list");
    }
    
    return *std::min_element(cones.begin(), cones.end(),
        [](const Cone& a, const Cone& b) {
            return a.distance < b.distance;
        });
}

double Point_To_Pixel_Node::calculateAngle(const Cone& from, const Cone& to) {
    return std::atan2(to.point.y - from.point.y, to.point.x - from.point.x);
}

std::vector<Cone> Point_To_Pixel_Node::orderConesByPathDirection(const std::vector<Cone>& unordered_cones) {
    if (unordered_cones.size() <= 1) {
        return unordered_cones;
    }

    std::vector<Cone> ordered_cones;
    std::vector<Cone> remaining_cones = unordered_cones;
    
    // Start with the closest cone to origin
    Cone current_cone = findClosestCone(remaining_cones);
    ordered_cones.push_back(current_cone);
    
    // Remove the first cone from remaining cones
    remaining_cones.erase(
        std::remove_if(remaining_cones.begin(), remaining_cones.end(),
            [&current_cone](const Cone& c) {
                return c.point.x == current_cone.point.x && 
                       c.point.y == current_cone.point.y;
            }), 
        remaining_cones.end());

    double prev_angle = std::atan2(current_cone.point.y, current_cone.point.x);

    while (!remaining_cones.empty()) {
        // Find next best cone based on distance and angle continuation
        auto next_cone_it = std::min_element(remaining_cones.begin(), remaining_cones.end(),
            [&](const Cone& a, const Cone& b) {
                double angle_a = calculateAngle(current_cone, a);
                double angle_b = calculateAngle(current_cone, b);
                
                // Calculate angle differences
                double angle_diff_a = std::abs(angle_a - prev_angle);
                double angle_diff_b = std::abs(angle_b - prev_angle);
                
                // Combine distance and angle criteria
                double score_a = 0.7 * (a.distance / current_cone.distance) + 
                               0.3 * angle_diff_a;
                double score_b = 0.7 * (b.distance / current_cone.distance) + 
                               0.3 * angle_diff_b;
                
                return score_a < score_b;
            });

        current_cone = *next_cone_it;
        ordered_cones.push_back(current_cone);
        prev_angle = calculateAngle(ordered_cones[ordered_cones.size()-2], current_cone);
        remaining_cones.erase(next_cone_it);
    }

    return ordered_cones;
}

// Topic callback definition
void Point_To_Pixel_Node::topic_callback(const interfaces::msg::PPMConeArray::SharedPtr msg)
{
  // Logging Actions
  #if TIMING
    auto start_time = high_resolution_clock::now();
    int64_t ms_time_since_lidar_2 = (this->get_clock()->now().nanoseconds() - msg->header.stamp.sec * 1e9 - msg->header.stamp.nanosec) / 1000;
  #endif
  RCLCPP_INFO(this->get_logger(), "Received message with %zu cones", msg->cone_array.size());

  // Message Definition
  interfaces::msg::ConeArray message = interfaces::msg::ConeArray();
  message.header = msg->header;
  message.orig_data_stamp = msg->header.stamp; 
  message.blue_cones = std::vector<geometry_msgs::msg::Point> {};
  message.yellow_cones = std::vector<geometry_msgs::msg::Point> {};
  message.orange_cones = std::vector<geometry_msgs::msg::Point> {};
  message.unknown_color_cones = std::vector<geometry_msgs::msg::Point> {};
  geometry_msgs::msg::Point point_msg;

  // Retrieve Camera Frame
  cv::Mat frameBGR_1 = this->getCameraFrame(msg->header.stamp);

  #if TIMING
    auto camera_time = high_resolution_clock::now();
  #endif

  #if YOLO
    // YOLO Pre-Processing
    cv::Mat blob;
    cv::dnn::blobFromImage(frameBGR_1, blob, 1 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    this->net.setInput(blob);

    // Forward blob through model
    std::vector<cv::Mat> outputs;
    this->net.forward(outputs, this->net.getUnconnectedOutLayersNames());

    // Isolate detection outputs
    cv::Mat detections = outputs[0];
  #endif

  // Timing Variables
  #if TIMING
    int transform_time = 0;
    int coloring_time = 0;
  #endif

  std::vector<Cone> unordered_yellow_cones;
  std::vector<Cone> unordered_blue_cones;

  // Iterate through all points in /cpp_cones message
  for (int i = 0; i < msg->cone_array.size(); i++){
    #if TIMING
      auto loop_start = high_resolution_clock::now();
    #endif
      
    // Transform Point
    Eigen::Vector2d pixel_1 = this->transform(msg->cone_array[i].cone_points[0]);

    #if TIMING
      // Time for transform
      auto loop_transform = high_resolution_clock::now();
      transform_time = transform_time + std::chrono::duration_cast<std::chrono::microseconds>(loop_transform - loop_start).count();
    #endif

    // Identify the color at the transformed image pixel
    #if !YOLO
      int cone_class = this->get_hsv_color(pixel_1, frameBGR_1);
    #else
      int cone_class = this->get_yolo_color(pixel_1, detections, frameBGR_1.cols, frameBGR_1.rows);
    #endif

    #if TIMING
      // Time for coloring
      auto loop_coloring = high_resolution_clock::now();
      coloring_time = transform_time + std::chrono::duration_cast<std::chrono::microseconds>(loop_coloring - loop_transform).count();
    #endif

    point_msg.x = msg->cone_array[i].cone_points[0].x;
    point_msg.y = msg->cone_array[i].cone_points[0].y;
    point_msg.z = 0.0;
  
    #if VIZ
    RCLCPP_INFO(this->get_logger(), "Cone: Color %d, 2D[ x:%lf y:%lf ] from 3D[ x:%lf y:%lf z:%lf ]",
                cone_class, pixel_1[0], pixel_1[1],
                msg->cone_array[i].cone_points[0].x, msg->cone_array[i].cone_points[0].y, msg->cone_array[i].cone_points[0].z);
    #endif

    switch (cone_class){
      case 0:
        #if VIZ
          std::cout << "orange\n" ;
        #endif
        message.orange_cones.push_back(point_msg);
        break;
      case 1:
        #if VIZ
          std::cout << "yellow\n" ;
        #endif
        unordered_yellow_cones.push_back(Cone(point_msg));
        break;
      case 2:
        #if VIZ
          std::cout << "blue\n" ;
        #endif
        unordered_blue_cones.push_back(Cone(point_msg));
        break;
      default:
        #if VIZ
          std::cout << "unknown\n";
        #endif
        message.unknown_color_cones.push_back(point_msg);
        break;
    }
  }

  #if TIMING
    auto transform_coloring_time = high_resolution_clock::now();
  #endif

  int cones_published = message.orange_cones.size() + message.yellow_cones.size() + message.blue_cones.size();
  int yellow_cones = message.yellow_cones.size();
  int blue_cones = message.blue_cones.size();
  int orange_cones = message.orange_cones.size();
  int unknown_color_cones = message.unknown_color_cones.size();
  
  RCLCPP_INFO(
    this->get_logger(), 
    "Transform callback triggered. Published %d cones. %d yellow, %d blue, %d orange, and %d unknown.", 
    cones_published, yellow_cones, blue_cones, orange_cones, unknown_color_cones
  );

  if (!unordered_yellow_cones.empty()) {
    std::vector<Cone> ordered_yellow = orderConesByPathDirection(unordered_yellow_cones);
    for (const auto& cone : ordered_yellow) {
      message.yellow_cones.push_back(cone.point);
    }
  }

  if (!unordered_blue_cones.empty()) {
    std::vector<Cone> ordered_blue = orderConesByPathDirection(unordered_blue_cones);
    for (const auto& cone : ordered_blue) {
      message.blue_cones.push_back(cone.point);
    }
  }

  #if TIMING
    auto end_time = high_resolution_clock::now();
    auto stamp_time = msg->header.stamp;
    auto ms_time_since_lidar = this->get_clock()->now() - stamp_time;

    RCLCPP_INFO(this->get_logger(), "Get Camera Frame  %ld microseconds.", std::chrono::duration_cast<std::chrono::microseconds>(camera_time - start_time).count());
    RCLCPP_INFO(this->get_logger(), "Total Transform and Coloring Time  %ld microseconds.", std::chrono::duration_cast<std::chrono::microseconds>(transform_coloring_time - camera_time).count());
    RCLCPP_INFO(this->get_logger(), "--Total Transform Time  %ld microseconds.", transform_time);
    RCLCPP_INFO(this->get_logger(), "--Total Coloring Time  %ld microseconds.", coloring_time);
    RCLCPP_INFO(this->get_logger(), "Total Transform and Coloring Time  %ld microseconds.", std::chrono::duration_cast<std::chrono::microseconds>(transform_coloring_time - camera_time));
    auto time_diff = end_time - start_time;
    RCLCPP_INFO(this->get_logger(), "Total PPM Time %ld microseconds.", std::chrono::duration_cast<std::chrono::microseconds>(time_diff).count());
    RCLCPP_INFO(this->get_logger(), "Total Time from Lidar  %ld microseconds.", ms_time_since_lidar.nanoseconds() / 1000);
    RCLCPP_INFO(this->get_logger(), "Total Time from Lidar to start  %ld microseconds.", ms_time_since_lidar_2);

  #endif
  
  this->publisher_->publish(message);
}

// void Point_To_Pixel_Node::twist_callback(const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
//   this->twist = msg;
// }

// void Point_To_Pixel_Node::quat_callback(const geometry_msgs::msg::QuaternionStamped::SharedPtr msg) {
//   this->quat = msg;
// }

// Camera Callback (Populates and maintain deque)
void Point_To_Pixel_Node::camera_callback()
{
  int MAX_DEQUE_SIZE = 10;
  this->frame_1 = this->cap_1.getLastFrame();

  cv::Mat frameBGR_1, left_raw, left_rect;

  // Format Frame
  if (frame_1.data != nullptr) {
    cv::Mat frameYUV_1 = cv::Mat(frame_1.height, frame_1.width, CV_8UC2, frame_1.data);
    cv::cvtColor(frameYUV_1,frameBGR_1,cv::COLOR_YUV2BGR_YUYV);
    left_raw = frameBGR_1(cv::Rect(0, 0, frameBGR_1.cols / 2, frameBGR_1.rows));
    
    // ----> Apply rectification
    cv::remap(left_raw, left_rect, this->map_left_x, this->map_left_y, cv::INTER_LINEAR);
    frameBGR_1 = left_rect;
  } else {
    RCLCPP_ERROR(this->get_logger(), "Failed to capture frame from camera 1.");
  }

  // Deque Management and Updating
  if (img_deque.size() < MAX_DEQUE_SIZE) {
    rclcpp::Time time = this->get_clock()->now();
    this->img_deque.push_back(std::make_pair(time, frameBGR_1));
  } else
  {
    this->img_deque.pop_front();
    this->img_deque.push_back(std::make_pair(this->get_clock()->now(), frameBGR_1));
  }
}

// First attempt to motion model the LiDAR points to the camera timestamp
interfaces::msg::PPMConeArray::SharedPtr Point_To_Pixel_Node::motion_model(
  Eigen::Vector3d LiDAR_linear,
  Eigen::Quaterniond LiDAR_quat,
  Eigen::Vector3d Camera_linear,
  Eigen::Quaterniond Camera_quat,
  rclcpp::Time timestamp_camera, 
  rclcpp::Time timestamp_lidar, 
  const interfaces::msg::PPMConeArray::SharedPtr LiDAR_cones) {
  // Get the time difference between the camera and lidar timestamps
  double time_diff = (timestamp_camera - timestamp_lidar).seconds();

  // Compute rotation matrix from quaternion to convert sensor frame points to global frame
  Eigen::Matrix3d get_sensor_to_global(Eigen::Quaterniond quat) {
    double q0 = quat.w();
    double q1 = quat.x(); 
    double q2 = quat.y();
    double q3 = quat.z();

    Eigen::Matrix3d rot_matrix;
    rot_matrix << 2*(q0*q0 + q1*q1) - 1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2),
                  2*(q1*q2 + q0*q3), 2*(q0*q0 + q2*q2) - 1, 2*(q2*q3 - q0*q1),
                  2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(q0*q0 + q3*q3) - 1;
    
    return rot_matrix;
  }

  // Compute rotation matrix to convert global frame points to sensor frame
  Eigen::Matrix3d get_global_to_sensor(Eigen::Quaterniond quat) {
    return get_sensor_to_global(quat).inverse();
  }

  // Compute translation between two timestamps using average velocity
  // dt is the time difference between the camera and lidar timestamps
  Eigen::Vector3d get_translation_to(Eigen::Vector3d LiDAR_linear, Eigen::Vector3d Camera_linear) {
    // average linear twist information between the two motion infos
    // multiply by dt to get translation
    return (LiDAR_linear + Camera_linear) * (time_diff / 2);
  }

  // Convert points to Eigen matrix format
  // This should be a 3xN matrix
  Eigen::MatrixXd points_matrix(3, LiDAR_cones->cone_array.size());
  for(size_t i = 0; i < LiDAR_cones->cone_array.size(); i++) {
    points_matrix.col(i) << LiDAR_cones->cone_array[i].cone_points[0].x, LiDAR_cones->cone_array[i].cone_points[0].y, LiDAR_cones->cone_array[i].cone_points[0].z;
  }

  // (1) Convert heading to global frame
  // This should be a 3x3 matrix
  Eigen::Matrix3d curr_sensor_to_global = get_sensor_to_global(LiDAR_quat);
  // This should be a 3xN matrix
  Eigen::MatrixXd curr_in_global = curr_sensor_to_global * points_matrix;

  // (2) Translate points to camera timestamp's global frame
  // This should be a 3x1 matrix
  Eigen::Vector3d translation = get_translation_to(LiDAR_linear, Camera_linear);
  // Broadcasting translation to all points by subtracting from each column
  // This should be a 3xN matrix
  Eigen::MatrixXd other_in_global = curr_in_global.colwise() - translation;

  // (3) Convert back to sensor frame at camera timestamp
  // This should be a 3x3 matrix
  Eigen::Matrix3d other_global_to_sensor = get_global_to_sensor(Camera_quat);
  // This should be a 3xN matrix
  Eigen::MatrixXd points_in_other = other_global_to_sensor * other_in_global;

  // Convert back to PPMConeArray message
  auto transformed_msg = std::make_shared<interfaces::msg::PPMConeArray>();
  transformed_msg->header = LiDAR_cones->header;
  transformed_msg->cone_array.resize(points_in_other.cols());

  for(size_t i = 0; i < points_in_other.cols(); i++) {
    transformed_msg->cone_array[i].cone_points.resize(1);
    transformed_msg->cone_array[i].cone_points[0].x = points_in_other.col(i)[0];
    transformed_msg->cone_array[i].cone_points[0].y = points_in_other.col(i)[1]; 
    transformed_msg->cone_array[i].cone_points[0].z = points_in_other.col(i)[2];
  }

  return transformed_msg;
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // Multithreading for timer callback
  rclcpp::executors::MultiThreadedExecutor executor;
  rclcpp::Node::SharedPtr node = std::make_shared<Point_To_Pixel_Node>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
