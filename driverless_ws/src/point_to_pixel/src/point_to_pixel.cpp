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


using namespace std::chrono_literals;
using std::placeholders::_1;

// ---------------------------------------------------------------------------
//    FLAGS
// ---------------------------------------------------------------------------
#define VIZ 1 // Prints color detection outputs of every point
#define VERBOSE 0 // Prints transform matrix and transformed pixel of every point
#define YOLO 0 // 0: HSV Coloring | 1: YOLO Coloring
#define TIMING 1 // Prints timing suite at end of every callback
#define INNER 1 // Uses inner lens of ZEDS (if 0 uses the outer lens)


class Point_To_Pixel_Node : public rclcpp::Node
{
  public:
    // Constructor declaration
    Point_To_Pixel_Node(); 

  private:
    // Image Deque
    std::deque<std::pair<rclcpp::Time, cv::Mat>> img_deque0;
    std::deque<std::pair<rclcpp::Time, cv::Mat>> img_deque1;

    // ROS2 Parameters
    Eigen::Matrix<double, 3, 4> projection_matrix0;
    Eigen::Matrix<double, 3, 4> projection_matrix1;

    double CONFIDENCE_THRESHOLD;
    cv::Scalar yellow_filter_high;
    cv::Scalar yellow_filter_low;
    cv::Scalar blue_filter_high;
    cv::Scalar blue_filter_low;
    cv::Scalar orange_filter_high;
    cv::Scalar orange_filter_low;


    // Topic Callback Functions
    void topic_callback(const interfaces::msg::PPMConeArray::SharedPtr msg); 
    std::pair<Eigen::Vector2d, Eigen::Vector2d> transform(geometry_msgs::msg::Vector3 &point);


    // Camera Callback Functions
    void camera_callback();
    rclcpp::TimerBase::SharedPtr camera_timer_;
    std::pair<cv::Mat, cv::Mat> get_camera_frame(rclcpp::Time callbackTime);

    // Coloring Functions
    #if YOLO
      cv::dnn::Net net; // YOLO Model
      std::pair<int, double> get_yolo_color(Eigen::Vector2d& pixel, cv::Mat image, int cols, int rows);
    #else
      std::pair<int, double> get_hsv_color(Eigen::Vector2d& pixel, cv::Mat image);
    #endif

    int get_cone_class(std::pair<Eigen::Vector2d, Eigen::Vector2d> pixel_pair,
                         std::pair<cv::Mat, cv::Mat> frame_pair,
                         std::pair<cv::Mat, cv::Mat> detection);

    // Camera Objects and Parameters
    sl_oc::video::VideoParams params;
    sl_oc::video::VideoCapture cap_0;
    sl_oc::video::VideoCapture cap_1;

    sl_oc::video::Frame canvas;
    sl_oc::video::Frame frame_0;
    sl_oc::video::Frame frame_1;

    cv::Mat map_left_x0, map_left_y0;
    cv::Mat map_right_x0, map_right_y0;

    cv::Mat map_left_x1, map_left_y1;
    cv::Mat map_right_x1, map_right_y1;

    // ROS2 Publisher and Subscribers
    rclcpp::Publisher<interfaces::msg::ConeArray>::SharedPtr publisher_;
    rclcpp::Subscription<interfaces::msg::PPMConeArray>::SharedPtr subscriber_;
};

// Constructor definition
Point_To_Pixel_Node::Point_To_Pixel_Node() : Node("point_to_pixel"),
                                              params([]() {sl_oc::video::VideoParams p; p.res = sl_oc::video::RESOLUTION::HD1080; p.fps = sl_oc::video::FPS::FPS_30; return p;}()),
                                              cap_0(sl_oc::video::VideoCapture(params)),
                                              cap_1(sl_oc::video::VideoCapture(params))
{
  // Camera 0
  if(!(this->cap_0).initializeVideo())
  {
    RCLCPP_ERROR(this->get_logger(), "Cannot open camera 0 video capture");
    rclcpp::shutdown(); // Shutdown node
  }
  RCLCPP_INFO(this->get_logger(), "Connected to ZED camera 0. %s", (this->cap_0).getDeviceName().c_str());

  // Camera 1
  if(!(this->cap_1).initializeVideo())
  {
    RCLCPP_ERROR(this->get_logger(), "Cannot open camera 1 video capture");
    rclcpp::shutdown(); // Shutdown node
  }
  RCLCPP_INFO(this->get_logger(), "Connected to ZED camera 1. %s", (this->cap_1).getDeviceName().c_str());

  // Retrieve calibration file from Stereolabs server
  int sn0 = this->cap_0.getSerialNumber();
  int sn1 = this->cap_1.getSerialNumber();

  std::string calibration_file0;
  std::string calibration_file1;

  unsigned int serial_number0 = sn0;
  unsigned int serial_number1 = sn1;

  // --> Download camera calibration file
  if (!sl_oc::tools::downloadCalibrationFile(serial_number0, calibration_file0))
  {
    std::cerr << "Could not load calibration file from Stereolabs servers for Camera 0" << std::endl;
  }
  if (!sl_oc::tools::downloadCalibrationFile(serial_number1, calibration_file1))
  {
    std::cerr << "Could not load calibration file from Stereolabs servers for Camera 1" << std::endl;
  }

  // --> Get Frame size
  int w0, h0;
  int w1, h1;
  this->cap_0.getFrameSize(w0, h0);
  this->cap_1.getFrameSize(w1, h1);
  cv::Mat cameraMatrix_left0, cameraMatrix_right0;
  cv::Mat cameraMatrix_left1, cameraMatrix_right1;

  // --> Initialize calibration
  sl_oc::tools::initCalibration(calibration_file1, cv::Size(w1 / 2, h1),
                                this->map_left_x0, this->map_left_y0, 
                                this->map_right_x0, this->map_right_y0,
                                cameraMatrix_left0, cameraMatrix_right0);
  sl_oc::tools::initCalibration(calibration_file1, cv::Size(w1 / 2, h1), 
                                this->map_left_x1, this->map_left_y1,
                                this->map_right_x1, this->map_right_y1,
                                cameraMatrix_left1, cameraMatrix_right1);


  // Set auto exposure and brightness
  this->cap_0.setAECAGC(true);
  this->cap_1.setAECAGC(true);

  RCLCPP_INFO(this->get_logger(), "ZED Camera 0 Ready. %s", (this->cap_0).getDeviceName().c_str());
  RCLCPP_INFO(this->get_logger(), "ZED Camera 1 Ready. %s", (this->cap_1).getDeviceName().c_str());

  // ---------------------------------------------------------------------------
  //                               PARAMETERS
  // ---------------------------------------------------------------------------

  // Initialize Empty IMG Deque
  this->img_deque0 = {};
  this->img_deque1 = {};

  // Projection matrix that takes LiDAR points to pixels
  std::vector<double> param_default0(12, 1.0f); 
  std::vector<double> param_default1(12, 1.0f);

  this->declare_parameter("projection_matrix0", param_default0);
  this->declare_parameter("projection_matrix1", param_default1);

  // Threshold that determines whether it reports the color on a cone or not
  this->declare_parameter("confidence_threshold", 0.05);

  #if YOLO
    // Load YOLO Model
    this->net = cv::dnn::readNetFromONNX("src/point_to_pixel/config/best164.onnx");
    if (this->net.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Error Loading YOLO Model");
      rclcpp::shutdown();
    }
    
    // Set CUDA On
    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  #endif

  // Default Color Parameters
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
  std::vector<double> param0 = this->get_parameter("projection_matrix0").as_double_array();
  std::vector<double> param1 = this->get_parameter("projection_matrix1").as_double_array();
  this->projection_matrix0 = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(param0.data());
  this->projection_matrix1 = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(param1.data());

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

  // std::chrono::seconds duration(3);
  // rclcpp::sleep_for(duration);

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

  // Camera Callback (25 fps)
  camera_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(40),
    [this](){this->camera_callback();}
  );

  // ---------------------------------------------------------------------------
  //                       INITIALIZATION COMPLETE SEQUENCE
  // ---------------------------------------------------------------------------
  
  // Retrieve freeze frame for calibration
  const sl_oc::video::Frame frame_0 = this->cap_0.getLastFrame();
  const sl_oc::video::Frame frame_1 = this->cap_1.getLastFrame();

  cv::Mat frameBGR_0, left_raw0, left_rect0, right_raw0, right_rect0;
  cv::Mat frameBGR_1, left_raw1, left_rect1, right_raw1, right_rect1;

  if (frame_0.data != nullptr)
  {
    cv::Mat frameYUV_0 = cv::Mat(frame_0.height, frame_0.width, CV_8UC2, frame_0.data);

    cv::cvtColor(frameYUV_0, frameBGR_0, cv::COLOR_YUV2BGR_YUYV);

    // ----> Extract left and right images from side-by-side
    left_raw0 = frameBGR_0(cv::Rect(0, 0, frameBGR_0.cols / 2, frameBGR_0.rows));
    right_raw0 = frameBGR_0(cv::Rect(frameBGR_0.cols / 2, 0, frameBGR_0.cols / 2, frameBGR_0.rows));

    // ----> Apply rectification
    cv::remap(left_raw0, left_rect0, this->map_left_x0, this->map_left_y0, cv::INTER_LINEAR);
    cv::remap(right_raw0, right_rect0, this->map_right_x0, this->map_right_y0, cv::INTER_LINEAR);
    frameBGR_0(cv::Rect(0, 0, frameBGR_0.cols / 2, frameBGR_0.rows)) = left_rect0;
    frameBGR_0(cv::Rect(frameBGR_0.cols / 2, 0, frameBGR_0.cols / 2, frameBGR_0.rows)) = right_rect0;
  }

  if (frame_1.data != nullptr)
  {
    cv::Mat frameYUV_1 = cv::Mat(frame_1.height, frame_1.width, CV_8UC2, frame_1.data);

    cv::cvtColor(frameYUV_1,frameBGR_1,cv::COLOR_YUV2BGR_YUYV);

    // ----> Extract left and right images from side-by-side
    left_raw1 = frameBGR_1(cv::Rect(0, 0, frameBGR_1.cols / 2, frameBGR_1.rows));
    right_raw1 = frameBGR_1(cv::Rect(frameBGR_1.cols / 2, 0, frameBGR_1.cols / 2, frameBGR_1.rows));

    // ----> Apply rectification
    cv::remap(left_raw1, left_rect1, this->map_left_x1, this->map_left_y1, cv::INTER_LINEAR);
    cv::remap(right_raw1, right_rect1, this->map_right_x1, this->map_right_y1, cv::INTER_LINEAR);
    frameBGR_1(cv::Rect(0, 0, frameBGR_1.cols / 2, frameBGR_1.rows)) = left_rect1;
    frameBGR_1(cv::Rect(frameBGR_1.cols / 2, 0, frameBGR_1.cols / 2, frameBGR_1.rows)) = right_rect1;
  }
  
  // Save freeze.png
  cv::imwrite("src/point_to_pixel/config/freeze0.png", left_rect0);
  cv::imwrite("src/point_to_pixel/config/freeze1.png", left_rect1);


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
  #if INNER
    RCLCPP_INFO(this->get_logger(), "Using Inner Cameras on ZEDs");
  #else
    RCLCPP_INFO(this->get_logger(), "Using Outer Cameras on ZEDs")
  #endif
};

// Maps 3D point to 2D Pixel
std::pair<Eigen::Vector2d, Eigen::Vector2d> Point_To_Pixel_Node::transform(geometry_msgs::msg::Vector3& point)
{

  #if VERBOSE
    // Create a stringstream to log the matrix
    std::stringstream ss0;
    std::stringstream ss1;

    // Iterate over the rows and columns of the matrix and format the output
    for (int i = 0; i < this->projection_matrix0.rows(); ++i){
      for (int j = 0; j < this->projection_matrix0.cols(); ++j){
        ss0 << this->projection_matrix0(i, j) << " ";
        ss1 << this->projection_matrix1(i, j) << " ";
      }
      ss0 << "\n";
      ss1 << "\n";
    }
    // Log the projection_matrix using ROS 2 logger
    RCLCPP_INFO(this->get_logger(), "Projection Matrix 0:\n%s", ss0.str().c_str());
    RCLCPP_INFO(this->get_logger(), "Projection Matrix 1:\n%s", ss1.str().c_str());
  #endif

  // Convert point from topic type (geometry_msgs/msg/Vector3) to Eigen Vector3d
  Eigen::Vector4d lidar_pt(point.x, point.y, point.z, 1.0);

  // Apply projection matrix to LiDAR point
  Eigen::Vector3d transformed0 = this->projection_matrix0 * lidar_pt;
  Eigen::Vector3d transformed1 = this->projection_matrix1 * lidar_pt;

  // Divide by z coordinate for Euclidean normalization
  Eigen::Vector2d pixel0 (transformed0(0)/transformed0(2), transformed0(1)/transformed0(2));
  Eigen::Vector2d pixel1 (transformed1(0)/transformed1(2), transformed1(1)/transformed1(2));

  return std::make_pair(pixel0, pixel1);
}


// Returns closest frame to callback time from both cameras
std::pair<cv::Mat, cv::Mat> Point_To_Pixel_Node::get_camera_frame(rclcpp::Time callbackTime)
{
  // Initialize Variables
  int64 bestDiff0 = INT64_MAX;
  int64 bestDiff1 = INT64_MAX;

  cv::Mat closestFrame0;
  cv::Mat closestFrame1;

  // Check if deque empty
  if (this->img_deque0.empty()) 
  {
    RCLCPP_ERROR(this->get_logger(), "Image deque is empty! Cannot find matching frame.");
    closestFrame0 = cv::Mat();
  }
  else
  {
    // Iterate through deque, simple best diff calculation
    for (const auto &frame : this->img_deque0)
    {
      int64 timeDiff = abs(frame.first.nanoseconds() - callbackTime.nanoseconds());

      if (timeDiff < bestDiff0)
      {
        closestFrame0 = frame.second;
        bestDiff0 = timeDiff;
      }
    }
  }

  if (this->img_deque1.empty())
  {
    RCLCPP_ERROR(this->get_logger(), "Image deque is empty! Cannot find matching frame.");
    closestFrame0 = cv::Mat();
  }
  else
  {
    // Iterate through deque, simple best diff calculation
    for (const auto &frame : this->img_deque1)
    {
      int64 timeDiff = abs(frame.first.nanoseconds() - callbackTime.nanoseconds());

      if (timeDiff < bestDiff1)
      {
        closestFrame1 = frame.second;
        bestDiff1 = timeDiff;
      }
    }
  }

  return std::make_pair(closestFrame0, closestFrame1);
}

#if YOLO
  // Checks if pixel in bounding box, returns bounding box with highest score
  std::pair<int, double> Point_To_Pixel_Node::get_yolo_color(Eigen::Vector2d& pixel, cv::Mat detection, int cols, int rows)
  {
    int x = static_cast<int>(pixel(0));
    int y = static_cast<int>(pixel(1));

    // Post Processing
    const float x_factor = cols / 640.0;
    const float y_factor = rows / 640.0;

    // Loop through all detection
    for (int i = 0; i < detection.rows; ++i) 
    {
      double confidence = detection.at<double>(i, 4);

      if (confidence >= this->CONFIDENCE_THRESHOLD)
      {
        // Get bounding box coordinates
        float cx = detection.at<float>(i, 0) * x_factor;
        float cy = detection.at<float>(i, 1) * y_factor;
        float width = detection.at<float>(i, 2) * x_factor;
        float height = detection.at<float>(i, 3) * y_factor;
        
        // Calculate the bounding box corners
        float left = cx - width/2;
        float top = cy - height/2;
        float right = cx + width/2;
        float bottom = cy + height/2;
        
        // If pixel is inside the bounding box
        if (left <= x && x <= right && top <= y && y <= bottom)
        {
          // Find the highest class score
          double max_class_score = 0;
          int class_id = -1;
          
          // Assuming class scores start after index 5 (++j)
          for (int j = 5; j < detection.cols; ++j) 
          {
            double class_score = detection.at<float>(i, j);
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
#else
  // Applies HSV Filter to image frame
  // Returns max of all colored pixels within ROI 
  std::pair<int, double> Point_To_Pixel_Node::get_hsv_color(Eigen::Vector2d& pixel, cv::Mat img)
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

    // Transformed point out of frame
    if (x_min >= x_max || y_min >= y_max) {
      return std::make_pair(-1, 1.0);
    }

    // Extract ROI and convert to HSV
    cv::Mat roi = img(cv::Range(y_min, y_max), cv::Range(x_min, x_max));
    cv::Mat hsv_roi;
    cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);

    // Define HSV color ranges
    std::pair<cv::Scalar, cv::Scalar> yellow_range = {this->yellow_filter_low, this->yellow_filter_high};

    std::pair<cv::Scalar, cv::Scalar> blue_range = {this->blue_filter_low, this->blue_filter_high};

    std::pair<cv::Scalar, cv::Scalar> orange_range = {this->orange_filter_low, this->orange_filter_high};


    // Create color masks
    cv::Mat yellow_mask = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
    cv::Mat blue_mask = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
    cv::Mat orange_mask = cv::Mat::zeros(hsv_roi.size(), CV_8UC1);
    cv::Mat temp_mask;

    // Apply color masks
    cv::inRange(hsv_roi, yellow_range.first, yellow_range.second, temp_mask);
    cv::bitwise_or(yellow_mask, temp_mask, yellow_mask);

    cv::inRange(hsv_roi, blue_range.first, blue_range.second, temp_mask);
    cv::bitwise_or(blue_mask, temp_mask, blue_mask);

    cv::inRange(hsv_roi, orange_range.first, orange_range.second, temp_mask);
    cv::bitwise_or(orange_mask, temp_mask, orange_mask);

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
        return std::make_pair(0, orange_percentage);
    } else if (yellow_percentage > this->CONFIDENCE_THRESHOLD && yellow_percentage > std::max(blue_percentage, orange_percentage) * RATIO_THRESHOLD) {
        return std::make_pair(1, yellow_percentage);
    } else if (blue_percentage > this->CONFIDENCE_THRESHOLD && blue_percentage > std::max(yellow_percentage, orange_percentage) * RATIO_THRESHOLD) {
        return std::make_pair(1, blue_percentage);
    }
    return std::make_pair(-1, 1.0);
  }
#endif

int Point_To_Pixel_Node::get_cone_class(std::pair<Eigen::Vector2d, Eigen::Vector2d> pixel_pair,
                                        std::pair<cv::Mat, cv::Mat> frame_pair,
                                        std::pair<cv::Mat, cv::Mat> detection_pair)
{
  // Declare the pixels in left (0) and right (1) camera space
  // CONE_CLASS [-1, 0, 1, 2], CONFIDENCE [0<--->1]
  std::pair<int, double> pixel0;
  std::pair<int, double> pixel1;

  // Identify the color at the transformed image pixel
  #if YOLO
    pixel0 = this->get_yolo_color(pixel_pair.first, detection_pair.first, frame_pair.first.cols, frame_pair.first.rows);
    pixel1 = this->get_yolo_color(pixel_pair.second, detection_pair.second, frame_pair.second.cols, frame_pair.second.rows);
  #else
    pixel0 = this->get_hsv_color(pixel_pair.first, frame_pair.first);
    pixel1 = this->get_hsv_color(pixel_pair.second, frame_pair.second);
  #endif

  // Logic for handling detection results

  // Return 0 if 1 did not detect color
  if (pixel0.first != -1 && pixel1.first == -1) {return pixel0.first;}
  // Return 1 if 0 did not detect color
  else if (pixel0.first == -1 && pixel1.first != -1) {return pixel1.first;}
  // Return result with highest confidence if both detect color
  else if (pixel0.first != -1 && pixel1.first != -1) {
    if (pixel0.second > pixel1.second) return pixel0.first;
    else return pixel1.first;
  }
  else{return -1;}
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
  std::pair<cv::Mat, cv::Mat> frame_pair = this->get_camera_frame(msg->header.stamp);

  #if TIMING
    auto camera_time = high_resolution_clock::now();
  #endif


  #if YOLO
    // YOLO Pre-Processing
    cv::Mat blob0;
    cv::Mat blob1;

    cv::dnn::blobFromImage(frameBGR_0, blob0, 1 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    cv::dnn::blobFromImage(frameBGR_1, blobget_cone
    this->net.setInput(blob1);

    // Forward blob through model
    std::vector<cv::Mat> outputs0;
    std::vector<cv::Mat> outputs1;

    this->net.forward(outputs0, this->net.getUnconnectedOutLayersNames());
    this->net.forward(outputs1, this->net.getUnconnectedOutLayersNames());

    // Isolate detection outputs
    cv::Mat detection0 = outputs0[0];
    cv::Mat detection1 = outputs1[0];
    std::pair<cv::Mat, cv::Mat> detection_pair = std::make_pair(detection0, detection1);
#else
    // Initialize empty matrix if not YOLO
    std::pair<cv::Mat, cv::Mat> detection_pair = std::make_pair(cv::Mat(), cv::Mat());
  #endif

  // Timing Variables
  #if TIMING
    int transform_time = 0;
    int coloring_time = 0;
  #endif

  // Iterate through all points in /cpp_cones message
  for (int i = 0; i < msg->cone_array.size(); i++){
    #if TIMING
      auto loop_start = high_resolution_clock::now();
    #endif
      
    // Transform Point
    std::pair<Eigen::Vector2d, Eigen::Vector2d> pixel_pair = this->transform(msg->cone_array[i].cone_points[0]);

    #if TIMING
      // Time for transform
      auto loop_transform = high_resolution_clock::now();
      transform_time = transform_time + std::chrono::duration_cast<std::chrono::microseconds>(loop_transform - loop_start).count();
    #endif

    int cone_class = this->get_cone_class(pixel_pair, frame_pair, detection_pair);

    #if TIMING
      // Time for coloring
      auto loop_coloring = high_resolution_clock::now();
      coloring_time = coloring_time + std::chrono::duration_cast<std::chrono::microseconds>(loop_coloring - loop_transform).count();
    #endif

    point_msg.x = msg->cone_array[i].cone_points[0].x;
    point_msg.y = msg->cone_array[i].cone_points[0].y;
    point_msg.z = 0.0;
  
    #if VIZ
    RCLCPP_INFO(this->get_logger(), "Cone: Color %d, 2D[ 0: (%lf, %lf) | 1: (%lf, %lf) ] from 3D[ (%lf, %lfl, %lf)",
                cone_class, pixel_pair.first[0], pixel_pair.first[1], pixel_pair.second[0], pixel_pair.second[1],
                msg->cone_array[i].cone_points[0].x, msg->cone_array[i].cone_points[0].y, msg->cone_array[i].cone_points[0].z);
#endif

    switch (cone_class){
      case 0:
        message.orange_cones.push_back(point_msg);
        break;
      case 1:
        message.yellow_cones.push_back(point_msg);
        break;
      case 2:
        message.blue_cones.push_back(point_msg);
        break;
      default:
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

  #if TIMING
    auto end_time = high_resolution_clock::now();
    auto stamp_time = msg->header.stamp;
    auto ms_time_since_lidar = this->get_clock()->now() - stamp_time;

    RCLCPP_INFO(this->get_logger(), "Get Camera Frame  %ld microseconds.", std::chrono::duration_cast<std::chrono::microseconds>(camera_time - start_time).count());
    RCLCPP_INFO(this->get_logger(), "Total Transform and Coloring Time  %ld microseconds.", std::chrono::duration_cast<std::chrono::microseconds>(transform_coloring_time - camera_time).count());
    RCLCPP_INFO(this->get_logger(), "--Total Transform Time  %ld microseconds.", transform_time);
    RCLCPP_INFO(this->get_logger(), "--Total Coloring Time  %ld microseconds.", coloring_time);
    auto time_diff = end_time - start_time;
    RCLCPP_INFO(this->get_logger(), "Total PPM Time %ld microseconds.", std::chrono::duration_cast<std::chrono::microseconds>(time_diff).count());
    RCLCPP_INFO(this->get_logger(), "Total Time from Lidar  %ld microseconds.", ms_time_since_lidar.nanoseconds() / 1000);
    RCLCPP_INFO(this->get_logger(), "Total Time from Lidar to start  %ld microseconds.", ms_time_since_lidar_2);

  #endif
  
  this->publisher_->publish(message);
}


// Camera Callback (Populates and maintain deque)
void Point_To_Pixel_Node::camera_callback()
{
  // Controls Max Deque Size
  int MAX_DEQUE_SIZE = 10;

  // Capture Frame and Time
  this->frame_0 = this->cap_1.getLastFrame();
  rclcpp::Time time0 = this->get_clock()->now();

  this->frame_1 = this->cap_1.getLastFrame();
  rclcpp::Time time1 = this->get_clock()->now();


  // Rectify Captured Frames
  cv::Mat frameBGR_0, left_raw0, left_rect0;
  cv::Mat frameBGR_1, left_raw1, left_rect1;

  if (frame_0.data != nullptr)
  {
    cv::Mat frameYUV_0 = cv::Mat(frame_0.height, frame_0.width, CV_8UC2, frame_0.data);
    cv::cvtColor(frameYUV_0, frameBGR_0, cv::COLOR_YUV2BGR_YUYV);
    left_raw0 = frameBGR_0(cv::Rect(0, 0, frameBGR_0.cols / 2, frameBGR_0.rows));

    // ----> Apply rectification
    cv::remap(left_raw0, left_rect0, this->map_left_x0, this->map_left_y0, cv::INTER_LINEAR);
    frameBGR_0 = left_rect0;
  }
  else
  {
    RCLCPP_ERROR(this->get_logger(), "Failed to capture frame from camera 0.");
  }

  if (frame_1.data != nullptr) {
    cv::Mat frameYUV_1 = cv::Mat(frame_1.height, frame_1.width, CV_8UC2, frame_1.data);
    cv::cvtColor(frameYUV_1,frameBGR_1,cv::COLOR_YUV2BGR_YUYV);
    left_raw1 = frameBGR_1(cv::Rect(0, 0, frameBGR_1.cols / 2, frameBGR_1.rows));
    
    // ----> Apply rectification
    cv::remap(left_raw1, left_rect1, this->map_left_x1, this->map_left_y1, cv::INTER_LINEAR);
    frameBGR_1 = left_rect1;
  } else {
    RCLCPP_ERROR(this->get_logger(), "Failed to capture frame from camera 1.");
  }

  // Deque Management and Updating
  if (img_deque0.size() >= MAX_DEQUE_SIZE) 
  {
    this->img_deque0.pop_front();
  }

  if (img_deque1.size() >= MAX_DEQUE_SIZE)
  {
    this->img_deque1.pop_front();
  }

  this->img_deque0.push_back(std::make_pair(time0, frameBGR_0));
  this->img_deque1.push_back(std::make_pair(time1, frameBGR_1));
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