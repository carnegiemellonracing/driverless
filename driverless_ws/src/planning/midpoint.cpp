#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int8.hpp"
#include "msg/cone_list.hpp"
#include "msg/points.hpp"
#include "generator.hpp"

using std::placeholders::_1;

// class qos_profile : public rclcpp::QoS{
    
// }

struct cones{

};

class MidpointNode : public rclcpp::Node
{
  private:
    void lap_callback(const std_msgs::msg::Int8::SharedPtr msg) 
    {
      lap=msg->data;
    }

    void cones_callback(const interfaces::msg::ConeList::SharedPtr msg) const
    { 
      if (lap>1) return;

      if((msg->blue_cones.size()==0 || msg->yellow_cones.size()==0) && (msg->orange_cones.size()<2)){
        return;
      }


      for (auto e : msg->blue_cones)
      {
        const std::pair<double, double> p = std::make_pair(e.x, e.y);
        std::vector<std::pair<double,double>>bluecones;
        bluecones.push_back(std::make_pair(e.x, e.y));
        perception_data.bluecones.push_back(std::make_pair(e.x, e.y));
      }      
      for (auto e : msg->orange_cones)
      {
        const std::pair<double, double> p = std::make_pair(e.x, e.y);
        perception_data.orangecones.push_back((std::pair<double, double>)p);
      }      
      for (auto e : msg->yellow_cones)
      {
        const std::pair<double, double> p = std::make_pair(e.x, e.y);
        perception_data.yellowcones.push_back((std::pair<double, double>)p);
      }
    }

    perceptionsData perception_data;

    rclcpp::Subscription<interfaces::msg::ConeList>::SharedPtr subscription_cones;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_lap_num;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr spline_publisher;

    int LOOKAHEAD_NEAR = 2;
    int LOOKAHEAD_FAR = 3;
    std::vector<double> vis_lookaheads;
    int lap = 1;
    bool vis_spline = true;
    
  public:
    MidpointNode()
    : Node("midpoint")
    {
      subscription_cones = this->create_subscription<interfaces::msg::ConeList>("/stereo_cones", 10, std::bind(&MidpointNode::cones_callback, this, _1));
      subscription_lap_num = this->create_subscription<std_msgs::msg::String>("/lap_num", 10, std::bind(&MidpointNode::lap_callback, this, _1));
      // VIS LOOKAHEADS
    }
};

