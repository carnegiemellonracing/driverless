// Open CV and Eigen
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

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

// ROS2 imports
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "interfaces/msg/ppm_cone_array.hpp"
#include "interfaces/msg/cone_list.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;

#define DEBUG 0

class Point_To_Pixel_Node : public rclcpp::Node
{
  public:
    Point_To_Pixel_Node(); // Constructor declaration

    static std::tuple<int, double> identify_color(Eigen::Vector2d& pixel, cv::Mat image);
    static void mouse_callback(int event, int x, int y, int flags, void* param);

  private:
    // Functions
    int transform(geometry_msgs::msg::Vector3& point); 

    // Parameters
    Eigen::Matrix<double, 3, 4> projection_matrix;

    // Callbacks/helper functions
    void topic_callback(const interfaces::msg::PPMConeArray::SharedPtr msg);
    

    void opencv_callback();

    // Parameters
    double CONFIDENCE_THRESHOLD;

    sl_oc::video::VideoParams params;
    sl_oc::video::VideoCapture cap_0;
    sl_oc::video::VideoCapture cap_1;

    sl_oc::video::Frame canvas;
    sl_oc::video::Frame frame_0;
    sl_oc::video::Frame frame_1;

    cv::Mat map_left_x, map_left_y;
    cv::Mat map_right_x, map_right_y;

    // ROS2 Objects
    rclcpp::Publisher<interfaces::msg::ConeList>::SharedPtr publisher_;
    rclcpp::Subscription<interfaces::msg::PPMConeArray>::SharedPtr subscriber_;
    rclcpp::TimerBase::SharedPtr opencv_timer_;

    
};

// TODO: FIX ZED static ID per this forum https://github.com/stereolabs/zed-ros-wrapper/issues/94

// Constructor definition
Point_To_Pixel_Node::Point_To_Pixel_Node() : Node("point_to_pixel"),
                                              params([]() {sl_oc::video::VideoParams p; p.res = sl_oc::video::RESOLUTION::HD1080; p.fps = sl_oc::video::FPS::FPS_30; return p;}()),
                                              cap_0(sl_oc::video::VideoCapture(params)),
                                              cap_1(sl_oc::video::VideoCapture(params))
{
  // params.res = sl_oc::video::RESOLUTION::HD720;
  // params.fps = sl_oc::video::FPS::FPS_60;
  // if( !(this->cap_0).initializeVideo(0) )
  // {
  //   RCLCPP_ERROR(this->get_logger(), "Cannot open camera 0 video capture");
  //   rclcpp::shutdown(); // Shutdown node
  // }
  
  // RCLCPP_INFO(this->get_logger(), "Connected to ZED camera. %s", (this->cap_0).getDeviceName().c_str());

  if(!(this->cap_1).initializeVideo(0))
  {
    RCLCPP_ERROR(this->get_logger(), "Cannot open camera 1 video capture");
    rclcpp::shutdown(); // Shutdown node
  }

  RCLCPP_INFO(this->get_logger(), "Connected to ZED camera. %s", (this->cap_1).getDeviceName().c_str());

  // Publisher that returns colored cones
  publisher_ = this->create_publisher<interfaces::msg::ConeList>("colored_cones", 10);
  
  // Subscriber that reads the input topic that contains an array of cone_point arrays from LiDAR stack
  subscriber_ = this->create_subscription<interfaces::msg::PPMConeArray>(
    "cone_array", 
    10, 
    [this](const interfaces::msg::PPMConeArray::SharedPtr msg) {this->topic_callback(msg);}
  );

  #if DEBUG
  // Timer for Opencv
  opencv_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(100),
    [this](){this->opencv_callback();}
  );
  #endif

  // Set Parameters
  std::vector<double> param_default(12, 1.0f); // Projection matrix that takes LiDAR points to pixels
  this->declare_parameter("projection_matrix", param_default);

  this->declare_parameter("confidence_threshold", .5); // Threshold that determines whether it reports the color on a cone or not

  // Get parameters
  std::vector<double> param = this->get_parameter("projection_matrix").as_double_array();
  this->projection_matrix = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(param.data());


  this->CONFIDENCE_THRESHOLD = this->get_parameter("confidence_threshold").as_double();

  std::chrono::seconds duration(2);
  rclcpp::sleep_for(duration);

  // TEST RECTIFY
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

  // END TEST

  const sl_oc::video::Frame frame_1 = this->cap_1.getLastFrame();

  cv::Mat frameBGR_1, left_raw, left_rect, right_raw, right_rect;
    if (frame_1.data != nullptr){
        // ----> Conversion from YUV 4:2:2 to BGR for visualization
        // cv::Mat frameYUV_1 = cv::Mat(1280, 720, CV_8UC2, frame_0.data);
        cv::Mat frameYUV_1 = cv::Mat(frame_1.height, frame_1.width, CV_8UC2, frame_1.data);
        // cv::Mat frameBGR_1;
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

    cv::imwrite("/home/chip/Documents/driverless/driverless_ws/src/point_to_pixel_test/config/freeze.png", left_rect);

  

  RCLCPP_INFO(this->get_logger(), "Point to Pixel Node INITIALIZED");

  // for opencv test
  // this->canvas = this->cap_0.getLastFrame();
};

// Point to pixel transform
// returns 0 for blue cone, 1 for yellow cone, and 2 for orange cone
int Point_To_Pixel_Node::transform(geometry_msgs::msg::Vector3& point)
{
  // Create a stringstream to log the matrix
        std::stringstream ss;

        // Iterate over the rows and columns of the matrix and format the output
        for (int i = 0; i < this->projection_matrix.rows(); ++i)
        {
            for (int j = 0; j < this->projection_matrix.cols(); ++j)
            {
                ss << this->projection_matrix(i, j) << " ";
            }
            ss << "\n";
        }

        // Log the projection_matrix using ROS 2 logger
        RCLCPP_INFO(this->get_logger(), "3x4 projection_matrix:\n%s", ss.str().c_str());
  #if !DEBUG
  // Convert point from topic type (geometry_msgs/msg/Vector3) to Eigen Vector3d
  Eigen::Vector4d lidar_pt(point.x, point.y, point.z, 1.0);

  // Apply projection matrix to LiDAR point
  Eigen::Vector3d transformed = this->projection_matrix * lidar_pt;

  // Divide by z coordinate for euclidean normalization
  Eigen::Vector2d pixel_1 (transformed(0)/transformed(2), transformed(1)/transformed(2));

  // const sl_oc::video::Frame frame_0 = this->cap_0.getLastFrame();

  // RCLCPP_INFO(this->get_logger(), "%d, %d \n", pixel_1(0), pixel_1(1));
  
  #endif

  #if DEBUG
  // Bypass the transform for testing
  Eigen::Vector2d pixel_1 (point.x, point.y);
  #endif

  // cv::Mat frameBGR_0;

  //   if (frame_0.data != nullptr){
  //       // ----> Conversion from YUV 4:2:2 to BGR for visualization
  //       // cv::Mat frameYUV_0 = cv::Mat(1280, 720, CV_8UC2, frame_0.data); 
  //       cv::Mat frameYUV_0 = cv::Mat(frame_0.height, frame_0.width, CV_8UC2, frame_0.data);
  //       // cv::Mat frameBGR_0;
  //       cv::cvtColor(frameYUV_0,frameBGR_0,cv::COLOR_YUV2BGR_YUYV);
  //       // <---- Conversion from YUV 4:2:2 to BGR for visualization);
  //   }
  const sl_oc::video::Frame frame_1 = this->cap_1.getLastFrame();


  cv::Mat frameBGR_1, left_raw, left_rect, right_raw, right_rect;
    if (frame_1.data != nullptr){
        // ----> Conversion from YUV 4:2:2 to BGR for visualization
        // cv::Mat frameYUV_1 = cv::Mat(1280, 720, CV_8UC2, frame_0.data);
        cv::Mat frameYUV_1 = cv::Mat(frame_1.height, frame_1.width, CV_8UC2, frame_1.data);
        // cv::Mat frameBGR_1;
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
        
        // frameBGR_1(cv::Rect(0, 0, frameBGR_1.cols / 2, frameBGR_1.rows)) = left_rect;
        // frameBGR_1(cv::Rect(frameBGR_1.cols / 2, 0, frameBGR_1.cols / 2, frameBGR_1.rows)) = right_rect;

        frameBGR_1 = left_rect;

    }

  // if (frame_0.data == nullptr) {
  //   RCLCPP_ERROR(this->get_logger(), "Failed to capture frame from camera 0.");
  //   return 0;
  // }
  if (frame_1.data == nullptr) {
      RCLCPP_ERROR(this->get_logger(), "Failed to capture frame from camera 1.");
      return 0;
  }

  std::tuple<int, float> ppm = this->identify_color(pixel_1, frameBGR_1);

  cv::drawMarker(frameBGR_1, cv::Point(pixel_1(0), pixel_1(1)), 'r');
  cv::imshow("Transformed Point", frameBGR_1);

  while (true) {
        int key = cv::waitKey(0); // Wait indefinitely for a key press
        if (key == 27) { // ASCII value for ESC key
            break; // Exit the loop if ESC is pressed
        }
    }

  cv::destroyAllWindows();
  
  RCLCPP_INFO(this->get_logger(), "x: %f, y: %f, color: %d, conf: %f", pixel_1(0), pixel_1(1), std::get<0>(ppm), std::get<1>(ppm));
  
  return std::get<0>(ppm);
}


std::tuple<int, double> Point_To_Pixel_Node::identify_color(Eigen::Vector2d& pixel, cv::Mat img)
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
    if (x_min >= x_max || y_min >= y_max) {
        return std::make_tuple(-1, 0.0);
    }


    // Extract ROI and convert to HSV
    cv::Mat roi = img(cv::Range(y_min, y_max), cv::Range(x_min, x_max));
    cv::Mat hsv_roi;
    cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);

    // Define HSV color ranges
    std::vector<std::pair<cv::Scalar, cv::Scalar>> yellow_ranges = {
        {cv::Scalar(18, 50, 50), cv::Scalar(35, 255, 255)},
        {cv::Scalar(22, 40, 40), cv::Scalar(38, 255, 255)},
        {cv::Scalar(25, 30, 30), cv::Scalar(35, 255, 255)}
    };
    std::vector<std::pair<cv::Scalar, cv::Scalar>> blue_ranges = {
        {cv::Scalar(100, 50, 50), cv::Scalar(130, 255, 255)},
        {cv::Scalar(110, 50, 50), cv::Scalar(130, 255, 255)},
        {cv::Scalar(90, 50, 50), cv::Scalar(110, 255, 255)},
        {cv::Scalar(105, 30, 30), cv::Scalar(125, 255, 255)}
    };
    std::vector<std::pair<cv::Scalar, cv::Scalar>> orange_ranges = {
        {cv::Scalar(0, 100, 100), cv::Scalar(15, 255, 255)},
        {cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255)},
        {cv::Scalar(5, 120, 120), cv::Scalar(15, 255, 255)}
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
    std::cout << "Yellow Percentage: " << yellow_percentage * 100 << "%" << std::endl;
    std::cout << "Blue Percentage: " << blue_percentage * 100 << "%" << std::endl;
    std::cout << "Orange Percentage: " << orange_percentage * 100 << "%" << std::endl;


    const double MIN_CONFIDENCE = 0.05;
    const double RATIO_THRESHOLD = 1.5;

    // Determine cone color
    if (orange_percentage > MIN_CONFIDENCE && orange_percentage > std::max(yellow_percentage, blue_percentage) * RATIO_THRESHOLD) {
        return std::make_tuple(0, orange_percentage);
    } else if (yellow_percentage > MIN_CONFIDENCE && yellow_percentage > std::max(blue_percentage, orange_percentage) * RATIO_THRESHOLD) {
        return std::make_tuple(1, yellow_percentage);
    } else if (blue_percentage > MIN_CONFIDENCE && blue_percentage > std::max(yellow_percentage, orange_percentage) * RATIO_THRESHOLD) {
        return std::make_tuple(2, blue_percentage);
    }
    return std::make_tuple(-1, std::max({yellow_percentage, blue_percentage, orange_percentage}));
}


// Topic callback definition
void Point_To_Pixel_Node::topic_callback(const interfaces::msg::PPMConeArray::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "Received message with %zu cones", msg->cone_array.size());

  interfaces::msg::ConeList message = interfaces::msg::ConeList();
  message.blue_cones = std::vector<geometry_msgs::msg::Point> {};
  message.yellow_cones = std::vector<geometry_msgs::msg::Point> {};
  message.orange_cones = std::vector<geometry_msgs::msg::Point> {};
  geometry_msgs::msg::Point point_msg;

  // std::cout << "msg->cone_array: " << typeid(msg->cone_array[0][0]).name() << std::endl;

  for (int i = 0; i < msg->cone_array.size(); i++){
    int cone_class = this->transform(msg->cone_array[i].cone_points[0]);

    point_msg.x = msg->cone_array[i].cone_points[0].x;
    point_msg.y = msg->cone_array[i].cone_points[0].y;
    point_msg.z = 0.0;

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
        break;
    }
    
  }
  
  RCLCPP_INFO(this->get_logger(), "Transform callback triggered.");

  
  publisher_->publish(message);
};

void Point_To_Pixel_Node::opencv_callback() {
  this->frame_0 = this->cap_1.getLastFrame();
  cv::Mat frameBGR;


  
  if (frame_0.data != nullptr){
    // RCLCPP_INFO(this->get_logger(), "inside");

      // ----> Conversion from YUV 4:2:2 to BGR for visualization
      // cv::Mat frameYUV_0 = cv::Mat(1280, 720, CV_8UC2, frame_0.data); 
      cv::Mat frameYUV_0 = cv::Mat(frame_0.height, frame_0.width, CV_8UC2, frame_0.data);
      // cv::Mat frameBGR_0;
      cv::cvtColor(frameYUV_0,frameBGR,cv::COLOR_YUV2BGR_YUYV);
      // <---- Conversion from YUV 4:2:2 to BGR for visualization
      // sl_oc::tools::showImage( "Stream RGB", frameBGR, sl_oc::video::RESOLUTION::HD720);
      // cv::imshow("Display Window", frameBGR);

      cv::Mat frame_1_resize;
      cv::resize(frameBGR, frame_1_resize, cv::Size(), 1., 1.);
      cv::Rect roi(0, 0, 1920, 1080);
      frame_1_resize = frame_1_resize(roi);
      cv::imshow("Display Window", frame_1_resize);
      cv::imwrite("/home/chip/Documents/driverless/driverless_ws/src/point_to_pixel_test/config/freeze.png", frame_1_resize);
  }
  // Create a pair of pointers to pass both image and canvas into the callback
  std::pair<cv::Mat*, cv::Mat*> params(&frameBGR, &frameBGR);

  // Set mouse callback with the parameters
  cv::setMouseCallback("Display Window", mouse_callback, &params);

  cv::waitKey(1);
}

void drawTransparentRectangle(cv::Mat& image, int x_min, int x_max, int y_min, int y_max, cv::Scalar color, double alpha) {
    // Create a transparent overlay (using a transparent color with alpha channel)
    cv::Mat overlay;
    image.copyTo(overlay); // Create a copy of the image to overlay the rectangle on

    // Draw a rectangle on the overlay with the desired transparency
    cv::rectangle(overlay, cv::Point(x_min, y_min), cv::Point(x_max, y_max), color, -1); // Green rectangle

    // Blend the rectangle onto the original image
    cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);
}

void Point_To_Pixel_Node::mouse_callback(int event, int x, int y, int flags, void* param){
    if (event == cv::EVENT_LBUTTONDOWN){
        // Unpack parameters: first is the original image, second is the canvas
        auto* params = static_cast<std::pair<cv::Mat*, cv::Mat*>*>(param);
        cv::Mat* image = params->first;
        cv::Mat* canvas = params->second;

        Eigen::Vector2d pix(x, y);
        std::tuple<int, double> out = Point_To_Pixel_Node::identify_color(pix, *image);  // Pass the original image to identify_color
        std::cout << std::get<0>(out) << std::endl << std::get<1>(out) << std::endl;
        std::cout << x << std::endl << y << std::endl;

        // Draw transparent rectangle around the ROI for the identified color
        int side_length = 25;
        int x_min_blue = static_cast<int>(pix(0)) - side_length;
        int x_max_blue = static_cast<int>(pix(0)) + side_length;
        int y_min_blue = static_cast<int>(pix(1)) - side_length;
        int y_max_blue = static_cast<int>(pix(1)) + side_length;

        cv::Scalar color;

        // Choose color based on the identified color
        switch (std::get<0>(out)) {
            case 0:
                color = cv::Scalar(0, 165, 255);  // Orange in BGR
                break;
            case 1:
                color = cv::Scalar(0, 255, 255);  // Yellow in BGR
                break;
            case 2:
                color = cv::Scalar(255, 0, 0);    // Blue in BGR
                break;
            case -1:
                color = cv::Scalar(0, 255, 0);    // Green (no cone detected)
                break;
        }

        // Modify the canvas with the rectangle
        drawTransparentRectangle(*canvas, x_min_blue, x_max_blue, y_min_blue, y_max_blue, color, 0.5); // 30% transparency

        // Display the updated canvas (not the original image)
        cv::imshow("Display Window", *canvas);
    }
}

// Main
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Point_To_Pixel_Node>());
  rclcpp::shutdown();
  return 0;
}
