#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "msg/cone_list.hpp"
using std::placeholders::_1;

// class qos_profile : public rclcpp::QoS{
    
// }


class MidpointNode : public rclcpp::Node
{
  private:
    void cones_callback(const std_msgs::msg::String::SharedPtr msg) const
    { 
      RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
    }
    void lap_callback(const std_msgs::msg::String::SharedPtr msg) const
    {
      RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
    }
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_cones;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_lap_num;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr spline_publisher;


    
  public:
    MidpointNode()
    : Node("midpoint")
    {
      subscription_cones = this->create_subscription<std_msgs::msg::String>("/stereo_cones", 10, std::bind(&MidpointNode::cones_callback, this, _1));
      subscription_lap_num = this->create_subscription<std_msgs::msg::String>("/lap_num", 10, std::bind(&MidpointNode::lap_callback, this, _1));
      
    }
};

