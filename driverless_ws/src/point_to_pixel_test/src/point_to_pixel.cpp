#include <Eigen/Dense>
#include <cstdio>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>
#include <tuple>
#include <string>

// ROS2 imports
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "interfaces/msg/ppm_cone_array.hpp"
#include "interfaces/msg/cone_list.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;


class Point_To_Pixel_Node : public rclcpp::Node
{
  public:
    Point_To_Pixel_Node(); // Constructor declaration

    // Functions
    int transform(geometry_msgs::msg::Vector3& point); 

    // Parameters
    Eigen::Matrix<double, 3, 4> projection_matrix;

  private:
    // Callbacks/helper functions
    void timer_callback();
    void topic_callback(const interfaces::msg::PPMConeArray::SharedPtr msg);
    std::tuple<int, double> identify_color(Eigen::Vector2d& pixel, Eigen::MatrixXd& image);

    // Parameters
    double CONFIDENCE_THRESHOLD;

    // ROS2 Objects
    rclcpp::Publisher<interfaces::msg::ConeList>::SharedPtr publisher_;
    rclcpp::Subscription<interfaces::msg::PPMConeArray>::SharedPtr subscriber_;
    rclcpp::TimerBase::SharedPtr test_publish_timer_;
};

// Constructor definition
Point_To_Pixel_Node::Point_To_Pixel_Node() : Node("point_to_pixel")
{
  // Publisher that returns colored cones
  publisher_ = this->create_publisher<interfaces::msg::ConeList>("colored_cones", 10);
  
  // Subscriber that reads the input topic that contains an array of cone_point arrays from LiDAR stack
  subscriber_ = this->create_subscription<interfaces::msg::PPMConeArray>(
    "cone_array", 
    10, 
    [this](const interfaces::msg::PPMConeArray::SharedPtr msg) {this->topic_callback(msg);}
  );

  // Set Parameters
  std::vector<double> param_default(12, 1.0f); // Projection matrix that takes LiDAR points to pixels
  this->declare_parameter("projection_matrix", param_default);

  this->declare_parameter("confidence_threshold", .5); // Threshold that determines whether it reports the color on a cone or not

  // Get parameters
  std::vector<double> param = this->get_parameter("projection_matrix").as_double_array();
  Eigen::Map<Eigen::Matrix<double, 3, 4>> projection_matrix(param.data());

  double CONFIDENCE_THRESHOLD = this->get_parameter("confidence_threshold").as_double();

  // Timer
  test_publish_timer_ = this->create_wall_timer(
    500ms,
    [this]() {&Point_To_Pixel_Node::timer_callback;} //TODO: confirm that lambda works the same as the line after
    //std::bind(&Point_To_Pixel_Node::timer_callback, this)
  );

  RCLCPP_INFO(this->get_logger(), "Point to Pixel Node INITIALIZED");
};


// Point to pixel transform
// returns 0 for blue cone, 1 for yellow cone, and 2 for orange cone
int Point_To_Pixel_Node::transform(geometry_msgs::msg::Vector3& point)
{
  // Convert point from topic type (geometry_msgs/msg/Vector3) to Eigen Vector3d
  Eigen::Vector4d lidar_pt(point.x, point.y, point.z, 1);

  // Apply projection matrix to LiDAR point
  Eigen::Vector3d transformed = this->projection_matrix * lidar_pt;

  // Divide by z coordinate for euclidean normalization
  Eigen::Vector2d pixel (transformed(0)/transformed(2), transformed(1)/transformed(2));

  // std::tuple<int, float> ppm = this->identify_color(pixel, );
  
  // if (std::get<1>(ppm) > this->CONFIDENCE_THRESHOLD){
  //   return std::get<0>(ppm);
  // }
  return -1;
}

std::tuple<int, double> Point_To_Pixel_Node::identify_color(Eigen::Vector2d& pixel, Eigen::MatrixXd& image)
{
  // random test value while the function remains unimplemented
  return std::make_tuple(0, 0.4);
}


// Topic callback definition
void Point_To_Pixel_Node::topic_callback(const interfaces::msg::PPMConeArray::SharedPtr msg)
{
  printf("read topic");
  RCLCPP_INFO(this->get_logger(), "Received message with %zu cones", msg->cone_array.size());
};


// Timer callback definition
void Point_To_Pixel_Node::timer_callback()
{
  geometry_msgs::msg::Point point_msg;
  point_msg.x = 0.0;
  point_msg.y = 1.0;
  point_msg.z = 0.0;

  interfaces::msg::ConeList message = interfaces::msg::ConeList();
  message.blue_cones = std::vector<geometry_msgs::msg::Point> {point_msg};
  message.yellow_cones = std::vector<geometry_msgs::msg::Point> {point_msg};
  publisher_->publish(message);
};


// Main
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Point_To_Pixel_Node>());
  rclcpp::shutdown();
  return 0;
}
