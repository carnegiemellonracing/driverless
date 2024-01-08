#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
// #include <sensor_msgs/image_encodings.h>
#include <sl/Camera.hpp>
#include <cv_bridge/cv_bridge.h>

class PublisherNode : public rclcpp::Node {
public:
  PublisherNode() : Node("publisher_node") {
    publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/zedsdk_left_color_image", 10);
    timer_ = this->create_wall_timer(std::chrono::milliseconds(50), std::bind(&PublisherNode::timer_callback, this));

    // init_params.camera_resolution = sl::RESOLUTION_HD720P;
    // init_params.depth_mode = sl::DEPTH_MODE::ULTRA;
    // init_params.coordinate_units = sl::UNIT::METER;
    // init_params.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;
    // init_params.camera_fps = 0;

    sl::ERROR_CODE err = zed.open(init_params);
    if (err != sl::ERROR_CODE::SUCCESS) {
        RCLCPP_INFO(this->get_logger(), "Failed to open ZED camera");
    }
  }

private:
  cv::Mat slMat2cvMat(sl::Mat& input) {
    // Mapping the sl::Mat data to cv::Mat
    return cv::Mat(input.getHeight(), input.getWidth(), CV_8UC4, input.getPtr<sl::uchar1>(sl::MEM::CPU));
  }

  int getOCVtype(sl::MAT_TYPE type) {
    int cv_type = -1;
    RCLCPP_INFO(this->get_logger(), "Type: %li", type);
    RCLCPP_INFO(this->get_logger(), "%li", sl::MAT_TYPE::F32_C1);
    RCLCPP_INFO(this->get_logger(), "%li", sl::MAT_TYPE::F32_C2);
    RCLCPP_INFO(this->get_logger(), "%li", sl::MAT_TYPE::F32_C3);
    RCLCPP_INFO(this->get_logger(), "%li", sl::MAT_TYPE::F32_C4);
    RCLCPP_INFO(this->get_logger(), "%li", sl::MAT_TYPE::U8_C1);
    RCLCPP_INFO(this->get_logger(), "%li", sl::MAT_TYPE::U8_C2);
    RCLCPP_INFO(this->get_logger(), "%li", sl::MAT_TYPE::U8_C3);
    RCLCPP_INFO(this->get_logger(), "%li", sl::MAT_TYPE::U8_C4);

    switch (type) {
        case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
        case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
        case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
        case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
        case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
        case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
        case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
        case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    return cv_type;
  }

  // cv::Mat slMat2cvMat(sl::Mat& input) {
  //   // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
  //   // cv::Mat and sl::Mat will share a single memory structure
  //   return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(sl::MEM::CPU));
  // }
  void timer_callback() {
    if (zed.grab() == sl::ERROR_CODE::SUCCESS) {

        // Retrieve left image
        zed.retrieveImage(left_image, sl::VIEW::LEFT);

        // Convert ZED image to OpenCV image
        cv::Mat cv_image = slMat2cvMat(left_image);  // slMat2cvMat is a helper function

        // Convert OpenCV image to ROS Image message using cv_bridge
        const std::string encoding = "bgra8";
        cv_bridge::CvImagePtr cv_ptr = std::make_shared<cv_bridge::CvImage>();
        cv_ptr->image = cv_image;
        cv_ptr->encoding = encoding;
        sensor_msgs::msg::Image::SharedPtr ros_image = cv_ptr->toImageMsg();

        RCLCPP_INFO(this->get_logger(), "Publishing");
        publisher_->publish(*ros_image.get());
    }
    else{
        RCLCPP_INFO(this->get_logger(), "Failed to retrieve camera frames");
    }
    
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  sl::Camera zed;
  sl::InitParameters init_params;
  sl::Mat left_image;

};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PublisherNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
