#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/PointCloud2.hpp"
#include "sensor_msgs/msg/Image.hpp"
#include "sensor_msgs/msg/NavSatFix.hpp"
#include "eufs_msgs/msg/ConeArrayWithCovariance.hpp"
using std::placeholders::_1;
class MinimalSubscriber : public rclcpp::Node
{
  public:
    MinimalSubscriber(): Node("data_collection")
    {
      subscription_ = this->create_subscription<std_msgs::msg::String>(
      "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
    }

  private:
    void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
    }

    message_filters::Cache<T> 
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr _lidar_sub;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _zed_left_sub;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr _zed_depth_sub;
    rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr _gps_sub;
    rclcpp::Subscription<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr _gt_cones_sub;

    message_filters::Cache<sensor_msgs::msg::PointCloud2> _lidar_cache(_lidar_sub, 10);
    message_filters::Cache<sensor_msgs::msg::Image> _zed_left_cache(_zed_left_sub, 10);
    message_filters::Cache<sensor_msgs::msg::PointCloud2> _zed_depth_cache(_zed_depth_sub, 10);
    message_filters::Cache<sensor_msgs::msg::NavSatFix> _gps_cache(_gps_sub, 10);
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}