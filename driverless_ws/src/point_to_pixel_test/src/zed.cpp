#include <cstdio>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include <Eigen/Dense>

// ROS2 imports
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

// zed-open-capture library header
#include <videocapture.hpp>

#include <iostream>
#include <iomanip>

#include <opencv2/opencv.hpp>

class Zed_Node : public rclcpp::Node
{
    public:
        Zed_Node(); // Constructor Declaration

        // sl_oc::video::VideoCapture cap_;
    private:
};


// Constructor Definition
Zed_Node::Zed_Node() : Node("zed_node")
{
    sl_oc::video::VideoParams params;
    params.res = sl_oc::video::RESOLUTION::HD720;
    params.fps = sl_oc::video::FPS::FPS_60;

    // cap_ = sl_oc::video::VideoCapture(params);
    sl_oc::video::VideoCapture cap_(params);

    if( !cap_.initializeVideo() )
    {
        RCLCPP_ERROR(this->get_logger(), "Cannot open camera video capture");
        rclcpp::shutdown(); // Shutdown node
    }


    RCLCPP_INFO(this->get_logger(), "Connected to ZED camera. %s", cap_.getDeviceName().c_str());

    RCLCPP_INFO(this->get_logger(), "Zed Node INITIALIZED");

    // Get last available frame
    const sl_oc::video::Frame frame = cap_.getLastFrame();

    if (frame.data != nullptr){

        // ----> Conversion from YUV 4:2:2 to BGR for visualization
        cv::Mat frameYUV = cv::Mat( frame.height, frame.width, CV_8UC2, frame.data );
        cv::Mat frameBGR;
        cv::cvtColor(frameYUV,frameBGR,cv::COLOR_YUV2BGR_YUYV);
        // <---- Conversion from YUV 4:2:2 to BGR for visualization

        // RCLCPP_INFO(this->get_logger(), "frame data type: %s", typeid(frame.data));
        cv::imshow("Display Window", frameBGR);
    }
}


// Main
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Zed_Node>());
  rclcpp::shutdown();
  return 0;
}
