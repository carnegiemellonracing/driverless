#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/string.hpp"
#include "eufs_msgs/msg/waypoint.hpp"
using std::placeholders::_1;

class MinimalSubscriber : public rclcpp::Node
{
  public:
    MinimalSubscriber()
    : Node("minimal_subscriber")
    {
      subscription_ = this->create_subscription<eufs_msgs::msg::Waypoint>(
      "topic", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));
    }

  private:
    void topic_callback(const eufs_msgs::msg::Waypoint::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "I heard:"); // '%s'", msg->data.c_str()
    }
    rclcpp::Subscription<eufs_msgs::msg::Waypoint>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}