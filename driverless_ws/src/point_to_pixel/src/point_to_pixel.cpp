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

// Flags for additional functionality
#define VIZ 1 // if 1 will output an additional topic without std_msgs/Header header
#define VERBOSE 0 // Prints outputs and transform matrix
#define YOLO 1 // Controls whether we use Yolo or not for coloring
#define TIMING 0

class Point_To_Pixel_Node : public rclcpp::Node
{
  public:
    Point_To_Pixel_Node(); // Constructor declaration

  private:
    // Image Deque
    std::deque<std::pair<rclcpp::Time, cv::Mat>> img_deque;

    #if YOLO
      // YOLO Model
      cv::dnn::Net yolo;
    #endif

    // ROS2 Parameters
    Eigen::Matrix<double, 3, 4> projection_matrix;
    double CONFIDENCE_THRESHOLD;
    cv::Scalar yellow_filter_high;
    cv::Scalar yellow_filter_low;
    cv::Scalar blue_filter_high;
    cv::Scalar blue_filter_low;
    cv::Scalar orange_filter_high;
    cv::Scalar orange_filter_low;


    // Callbacks / Transform functions
    void topic_callback(const interfaces::msg::PPMConeArray::SharedPtr msg);
    void camera_callback();
    int transform(geometry_msgs::msg::Vector3& point, rclcpp::Time callbackTimer); 

    // Coloring Functions
    #if YOLO
    std::pair<int, double> get_yolo_color(Eigen::Vector2d& pixel, cv::Mat image);
    #endif
    std::pair<int, double> get_hsv_color(Eigen::Vector2d& pixel, cv::Mat image);
    
    cv::Mat getCameraFrame(rclcpp::Time callbackTime);

    // Parameters
    sl_oc::video::VideoParams params;
    sl_oc::video::VideoCapture cap_0;
    sl_oc::video::VideoCapture cap_1;

    sl_oc::video::Frame canvas;
    sl_oc::video::Frame frame_0;
    sl_oc::video::Frame frame_1;

    cv::Mat map_left_x, map_left_y;
    cv::Mat map_right_x, map_right_y;

    // ROS2 Objects
    rclcpp::Publisher<interfaces::msg::ConeArray>::SharedPtr publisher_;
    rclcpp::Subscription<interfaces::msg::PPMConeArray>::SharedPtr subscriber_;

    // Camera Callback(10 frames per second)
    rclcpp::TimerBase::SharedPtr camera_timer_;
    
};

// TODO: FIX ZED static ID per this forum https://github.com/stereolabs/zed-ros-wrapper/issues/94

// Constructor definition
Point_To_Pixel_Node::Point_To_Pixel_Node() : Node("point_to_pixel"),
                                              params([]() {sl_oc::video::VideoParams p; p.res = sl_oc::video::RESOLUTION::HD1080; p.fps = sl_oc::video::FPS::FPS_30; return p;}()),
                                              cap_0(sl_oc::video::VideoCapture(params)),
                                              cap_1(sl_oc::video::VideoCapture(params))
{
  // ---------------------------------------------------------------------------
  //    CONFIRM CAMERAS ARE ACCESSIBLE AND PROPERLY DECLARED AND INITIALIZED
  // ---------------------------------------------------------------------------

  // // Checks if the video capture object was properly declared and initialized
  // if(!(this->cap_0).initializeVideo(0))
  // {
  //   RCLCPP_ERROR(this->get_logger(), "Cannot open camera 1 video capture");
  //   rclcpp::shutdown(); // Shutdown node
  // }

  // RCLCPP_INFO(this->get_logger(), "Connected to ZED camera. %s", (this->cap_0).getDeviceName().c_str());

  // Checks if the video capture object was properly declared and initialized
  if(!(this->cap_1).initializeVideo(0))
  {
    RCLCPP_ERROR(this->get_logger(), "Cannot open camera 1 video capture");
    rclcpp::shutdown(); // Shutdown node
  }

  RCLCPP_INFO(this->get_logger(), "Connected to ZED camera. %s", (this->cap_1).getDeviceName().c_str());

  // ---------------------------------------------------------------------------
  //                           ROS2 PARAMETER INIT
  // ---------------------------------------------------------------------------

  this->img_deque = {};

  // Set Parameters

  // Projection matrix that takes LiDAR points to pixels
  std::vector<double> param_default(12, 1.0f); 
  this->declare_parameter("projection_matrix", param_default);

  // Threshold that determines whether it reports the color on a cone or not
  this->declare_parameter("confidence_threshold", .15); 

  #if YOLO
    // YOLO Stuff
    this->yolo = cv::dnn::readNetFromONNX("src/point_to_pixel/config/best164.onnx");

    if (this->yolo.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Error Loading YOLO Model");
      rclcpp::shutdown();
    }

    RCLCPP_INFO(this->get_logger(), "YOLO Model %s", (this->cap_1).getDeviceName().c_str());
    
    // Change preferable backend if we want to run on GPU
    this->yolo.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->yolo.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  #endif

  // **TO BE IMPLEMENTED **
  // Deque size and refresh rate 
  // Single or dual camera mode
  // Include calibration? 

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

  // Get parameters

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

  // ---------------------------------------------------------------------------
  //                            ROS2 OBJECTS SETUP
  // ---------------------------------------------------------------------------
  
  // Publisher that returns colored cones
  publisher_ = this->create_publisher<interfaces::msg::ConeArray>("/perc_cones", 10);
  
  // Subscriber that reads the input topic that contains an array of cone_point arrays from LiDAR stack
  subscriber_ = this->create_subscription<interfaces::msg::PPMConeArray>(
    "/cpp_cones", 
    10, 
    [this](const interfaces::msg::PPMConeArray::SharedPtr msg) {this->topic_callback(msg);}
  );

  // Camera Callback (10 fps)
  camera_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(10),
    [this](){this->camera_callback();}
  );
  
  // ---------------------------------------------------------------------------
  //                  INITIALIZE CAMERA RECTIFICATION MATRIX
  // ---------------------------------------------------------------------------
  
  // Rectify Image
  int sn = this->cap_1.getSerialNumber();

  // ----> Retrieve calibration file from Stereolabs server
  std::string calibration_file;
  // ZED Calibration
  unsigned int serial_number = sn;
  // Download camera calibration file
  if( !sl_oc::tools::downloadCalibrationFile(serial_number, calibration_file) )
  {
      std::cerr << "Could not load calibration file from Stereolabs servers" << std::endl;
  }
  // ----> Frame size
  int w,h;
  this->cap_1.getFrameSize(w,h);
  // <---- Frame size
  // ----> Initialize calibration
  cv::Mat cameraMatrix_left, cameraMatrix_right;
  sl_oc::tools::initCalibration(calibration_file, cv::Size(w/2,h), this->map_left_x, this->map_left_y, this->map_right_x, this->map_right_y,
                                cameraMatrix_left, cameraMatrix_right);

  // ---------------------------------------------------------------------------
  //               GRAB ONE FREEZE FRAME FOR CALIBRATION.PY SCRIPT
  // ---------------------------------------------------------------------------

  const sl_oc::video::Frame frame_1 = this->cap_1.getLastFrame();

  cv::Mat frameBGR_1, left_raw, left_rect, right_raw, right_rect;
  if (frame_1.data != nullptr){
      // ----> Conversion from YUV 4:2:2 to BGR for visualization
      // cv::Mat frameYUV_0 = cv::Mat(frame_0.height, frame_0.width, CV_8UC2, frame_0.data);
      cv::Mat frameYUV_1 = cv::Mat(frame_1.height, frame_1.width, CV_8UC2, frame_1.data);

      cv::cvtColor(frameYUV_1,frameBGR_1,cv::COLOR_YUV2BGR_YUYV);
      // <---- Conversion from YUV 4:2:2 to BGR for visualization
      // cv::Rect roi(0, 0, 1280, 720);
      // frameBGR_1 = frameBGR_1(roi);

      // ----> Extract left and right images from side-by-side
      left_raw = frameBGR_1(cv::Rect(0, 0, frameBGR_1.cols / 2, frameBGR_1.rows));
      right_raw = frameBGR_1(cv::Rect(frameBGR_1.cols / 2, 0, frameBGR_1.cols / 2, frameBGR_1.rows));

      // ----> Apply rectification
      cv::remap(left_raw, left_rect, this->map_left_x, this->map_left_y, cv::INTER_LINEAR );
      cv::remap(right_raw, right_rect, this->map_right_x, this->map_right_y, cv::INTER_LINEAR );
      frameBGR_1(cv::Rect(0, 0, frameBGR_1.cols / 2, frameBGR_1.rows)) = left_rect;
      frameBGR_1(cv::Rect(frameBGR_1.cols / 2, 0, frameBGR_1.cols / 2, frameBGR_1.rows)) = right_rect;
      
  }

  cv::imwrite("/home/chip/Documents/driverless/driverless_ws/src/point_to_pixel/config/freeze.png", left_rect);

  // Initialization complete msg
  RCLCPP_INFO(this->get_logger(), "Point to Pixel Node INITIALIZED");
};

// Point to Pixel coordinate transform
// returns 0 for blue cone, 1 for yellow cone, and 2 for orange cone
int Point_To_Pixel_Node::transform(
  geometry_msgs::msg::Vector3& point, 
  rclcpp::Time callbackTime
  )
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

  // Get camera frame that is closest to time of LiDAR point
  cv::Mat frameBGR_1 = this->getCameraFrame(callbackTime);

  // Identify the color at the transformed image pixel
  #if !YOLO
    std::pair<int, float> ppm = this->get_hsv_color(pixel_1, frameBGR_1);
  #else
    std::pair<int, float> ppm = this->get_yolo_color(pixel_1, frameBGR_1);
  #endif

  
  // RCLCPP_INFO(this->get_logger(), "x: %f, y: %f, color: %d, conf: %f", pixel_1(0), pixel_1(1), std::get<0>(ppm), std::get<1>(ppm));

  return ppm.first;
}


// Returns closest frame to callback time
cv::Mat Point_To_Pixel_Node::getCameraFrame(rclcpp::Time callbackTime)
{
  // Set as int max
  int64 bestDiff = INT64_MAX;
  // Returned Frame
  cv::Mat closestFrame;
  
  // For debugging
  int index = 0;
  int bestFrameIndex = 0;

  #if VERBOSE
    RCLCPP_INFO(this->get_logger(), "getCameraFrame called with time: %ld", 
                                  callbackTime.nanoseconds());
  #endif
  
  // If Deque is empty we have nothing to transform to
  if (this->img_deque.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Image deque is empty! Cannot find matching frame.");
    return cv::Mat(); // Return empty matrix
  }

  // Iterate through deque
  for (const auto& frame: this->img_deque) {
    int64 timeDiff = abs(frame.first.nanoseconds() - callbackTime.nanoseconds());

    if (timeDiff < bestDiff) {
      closestFrame = frame.second;
      bestDiff = timeDiff;
      bestFrameIndex = index;
    }
    index++;
  }
  
  // Set threshold
  if (closestFrame.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Could not get frame");
    return cv::Mat();
  }
  
  #if VERBOSE
    RCLCPP_INFO(this->get_logger(), "Best frame:%d | REQ-FRAME Time diff: %ld nanoseconds", 
              bestFrameIndex, bestDiff);
  #endif
  
  return closestFrame;
}

#if VIZ
  std::string pairToString(std::pair<cv::Scalar, cv::Scalar>& p) {
    std::stringstream ss;
    ss << "([" << std::to_string(p.first[0]) << ", " << std::to_string(p.first[1])<< ", " << std::to_string(p.first[2]) << "], [" << std::to_string(p.second[0]) <<  ", " << std::to_string(p.second[1]) << ", " << std::to_string(p.second[2]) << "])";
    return ss.str();
  }
#endif

#if YOLO
  std::pair<int, double> Point_To_Pixel_Node::get_yolo_color(Eigen::Vector2d& pixel, cv::Mat img)
  {
    int x = static_cast<int>(pixel(0));
    int y = static_cast<int>(pixel(1));

    // Pre-Processing 
    cv::Mat blob;
    cv::dnn::blobFromImage(img, blob, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    this->yolo.setInput(blob);

    std::vector<cv::Mat> outputs;
    this->yolo.forward(outputs, this->yolo.getUnconnectedOutLayersNames());

    // Post Processing
    const float x_factor = img.cols / 640.0;
    const float y_factor = img.rows / 640.0;
    
    // Process detections
    cv::Mat detections = outputs[0];
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


// Identifies Color from a camera pixel
std::pair<int, double> Point_To_Pixel_Node::get_hsv_color(Eigen::Vector2d& pixel, cv::Mat img)
{
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
    std::pair<cv::Scalar, cv::Scalar> _y = {this->yellow_filter_low, this->yellow_filter_high};
    std::cout << pairToString(_y) << std::endl;
    RCLCPP_INFO(this->get_logger(), "point out of frame? (y_min >= y_max): %d, %d, %s", y_min, y_max, y_min >= y_max ? "true": "false");
    RCLCPP_INFO(this->get_logger(), "point out of frame? (x_min >= x_max): %d, %d, %s", x_min, x_max, x_min >= x_max ? "true": "false");
  #endif

  

  if (x_min >= x_max || y_min >= y_max) {
      return std::make_pair(-1, 0.0);
  }

  // Extract ROI and convert to HSV
  cv::Mat roi = img(cv::Range(y_min, y_max), cv::Range(x_min, x_max));
  cv::Mat hsv_roi;
  cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);

  // Define HSV color ranges
  std::vector<std::pair<cv::Scalar, cv::Scalar>> yellow_ranges = {
    {this->yellow_filter_low, this->yellow_filter_high}
  };
  // {
  //     {cv::Scalar(18, 50, 50), cv::Scalar(35, 255, 255)},
  //     {cv::Scalar(22, 40, 40), cv::Scalar(38, 255, 255)},
  //     {cv::Scalar(25, 30, 30), cv::Scalar(35, 255, 255)}
  // };
  std::vector<std::pair<cv::Scalar, cv::Scalar>> blue_ranges = {
    {this->blue_filter_low, this->blue_filter_high}
  };
  // {
  //     {cv::Scalar(100, 50, 50), cv::Scalar(130, 255, 255)},
  //     {cv::Scalar(110, 50, 50), cv::Scalar(130, 255, 255)},
  //     {cv::Scalar(90, 50, 50), cv::Scalar(110, 255, 255)},
  //     {cv::Scalar(105, 30, 30), cv::Scalar(125, 255, 255)}
  // };
  std::vector<std::pair<cv::Scalar, cv::Scalar>> orange_ranges = {
    {this->orange_filter_low, this->orange_filter_high}
  };
  // {
  //     {cv::Scalar(0, 100, 100), cv::Scalar(15, 255, 255)},
  //     {cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255)},
  //     {cv::Scalar(5, 120, 120), cv::Scalar(15, 255, 255)}
  // };

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
  std::cout << "Yellow Percentage: " << yellow_percentage * 100 << "%" << std::endl;
  std::cout << "Blue Percentage: " << blue_percentage * 100 << "%" << std::endl;
  std::cout << "Orange Percentage: " << orange_percentage * 100 << "%" << std::endl;
  const double MIN_CONFIDENCE = 0.05;
  const double RATIO_THRESHOLD = 1.5;

  // Determine cone color
  if (orange_percentage > MIN_CONFIDENCE && orange_percentage > std::max(yellow_percentage, blue_percentage) * RATIO_THRESHOLD) {
      return std::make_pair(0, orange_percentage);
  } else if (yellow_percentage > MIN_CONFIDENCE && yellow_percentage > std::max(blue_percentage, orange_percentage) * RATIO_THRESHOLD) {
      return std::make_pair(1, yellow_percentage);
  } else if (blue_percentage > MIN_CONFIDENCE && blue_percentage > std::max(yellow_percentage, orange_percentage) * RATIO_THRESHOLD) {
      return std::make_pair(2, blue_percentage);
  }
  return std::make_pair(-1, std::max({yellow_percentage, blue_percentage, orange_percentage}));
}


// Topic callback definition
void Point_To_Pixel_Node::topic_callback(const interfaces::msg::PPMConeArray::SharedPtr msg)
{
  #if TIMING
    auto start_time = high_resolution_clock::now();
  #endif

  RCLCPP_INFO(this->get_logger(), "Received message with %zu cones", msg->cone_array.size());

  interfaces::msg::ConeArray message = interfaces::msg::ConeArray();
  message.header = msg->header;
  message.orig_data_stamp = msg->header.stamp; 
  message.blue_cones = std::vector<geometry_msgs::msg::Point> {};
  message.yellow_cones = std::vector<geometry_msgs::msg::Point> {};
  message.orange_cones = std::vector<geometry_msgs::msg::Point> {};
  message.unknown_color_cones = std::vector<geometry_msgs::msg::Point> {};
  geometry_msgs::msg::Point point_msg;
  
  for (int i = 0; i < msg->cone_array.size(); i++){
    int cone_class = this->transform(msg->cone_array[i].cone_points[0], msg->header.stamp);

    point_msg.x = msg->cone_array[i].cone_points[0].x;
    point_msg.y = msg->cone_array[i].cone_points[0].y;
    point_msg.z = 0.0;
  
    RCLCPP_INFO(this->get_logger(), "%d", cone_class);

    switch (cone_class){
      case 0:
        #if VERBOSE
          std::cout << "orange\n" ;
        #endif
        message.orange_cones.push_back(point_msg);
        break;
      case 1:
        #if VERBOSE
          std::cout << "yellow\n" ;
        #endif
        message.yellow_cones.push_back(point_msg);
        break;
      case 2:
        #if VERBOSE
          std::cout << "blue\n" ;
        #endif
        message.blue_cones.push_back(point_msg);
        break;
      default:
        message.unknown_color_cones.push_back(point_msg);
        break;
    }
  }

  #if VERBOSE
    int cones_published = message.orange_cones.size() + message.yellow_cones.size() + message.blue_cones.size();
    int yellow_cones = message.yellow_cones.size();
    int blue_cones = message.blue_cones.size();
    int orange_cones = message.orange_cones.size();
    
    RCLCPP_INFO(
      this->get_logger(), 
      "Transform callback triggered. Published %d cones. %d yellow, %d blue, and %d orange.", 
      cones_published, yellow_cones, blue_cones, orange_cones
    );
  #else
    RCLCPP_INFO(this->get_logger(), "Transform callback triggered");
  #endif

  #if TIMING
    auto end_time = high_resolution_clock::now();
    auto stamp_time = msg->header.stamp.nanoseconds() * cmath::pow(10, 3);

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(end_time - start_time);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = end_time - start_time;
    duration<double, std::milli> ms_time_since_lidar = end_time - stamp_time;


    RCLCPP_INFO(this->get_logger(), "Time from start to end of callback. as int: %u ms. as double: %lf ms", ms_int, ms_double);
    RCLCPP_INFO(this->get_logger(), "Time from start to end of callback. as int: %u ms. as double: %lf ms", ms_int, ms_double);
    

  #endif
  
  this->publisher_->publish(message);
}


// Camera Callback (Populates and maintain deque)
void Point_To_Pixel_Node::camera_callback()
{
  this->frame_1 = this->cap_1.getLastFrame();

  cv::Mat frameBGR_1, left_raw, left_rect; //, right_raw, right_rect;

  if (frame_1.data != nullptr){
    cv::Mat frameYUV_1 = cv::Mat(frame_1.height, frame_1.width, CV_8UC2, frame_1.data);
    cv::cvtColor(frameYUV_1,frameBGR_1,cv::COLOR_YUV2BGR_YUYV);
    left_raw = frameBGR_1(cv::Rect(0, 0, frameBGR_1.cols / 2, frameBGR_1.rows));
    // right_raw = frameBGR_1(cv::Rect(frameBGR_1.cols / 2, 0, frameBGR_1.cols / 2, frameBGR_1.rows));
    
    // ----> Apply rectification
    cv::remap(left_raw, left_rect, this->map_left_x, this->map_left_y, cv::INTER_LINEAR);
    // cv::remap(right_raw, right_rect, this->map_right_x, this->map_right_y, cv::INTER_LINEAR);
    frameBGR_1 = left_rect;
  } else {
    RCLCPP_ERROR(this->get_logger(), "Failed to capture frame from camera 1.");
  }

  // Deque Management and Updating
  if (img_deque.size() < 10) {
    rclcpp::Time time = this->get_clock()->now();
    this->img_deque.push_back(std::make_pair(time, frameBGR_1));
  } else
  {
    this->img_deque.pop_front();
    this->img_deque.push_back(std::make_pair(this->get_clock()->now(), frameBGR_1));
  }
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Point_To_Pixel_Node>());
  rclcpp::shutdown();
  return 0;
}
