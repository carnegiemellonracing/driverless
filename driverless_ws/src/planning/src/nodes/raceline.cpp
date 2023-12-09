#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int8.hpp"
#include "geometry_msgs/msg/point.hpp"
// #include "msg/optimizer_points.hpp"
#include "eufs_msgs/msg/cone_array.hpp"
#include "eufs_msgs/msg/point_array.hpp"
// #include "interfaces/msg/cone_list.hpp"
// #include "interfaces/msg/points.hpp"
#include "../planning_codebase/midline/generator.hpp"
#include "interfaces/msg/spline.hpp"
// #include "frenet.hpp"
// #include "runpy.hpp"
#include <Eigen/Dense>


static const std::string CONES_TOPIC = "/slam_cones";
static const std::string RACELINE_TOPIC = "/raceline_splines";

using std::placeholders::_1;
#define DELTA 0.5
struct raceline_pt{
  double x,y,w_l,w_r;
};



class RacelineNode : public rclcpp::Node
{
  private:
    perceptionsData perception_data;

    rclcpp::Subscription<eufs_msgs::msg::ConeArray>::SharedPtr subscription_cones;
    // rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_lap_num;
    // rclcpp::Publisher<eufs_msgs::msg::PointArray>::SharedPtr publisher_rcl_pt;
    rclcpp::Publisher<interfaces::msg::Spline>::SharedPtr publisher_rcl_pt;

    std::vector<double> vis_lookaheads;
    int lap = 1;
    bool vis_spline = true;
    MidpointGenerator generator_mid;

    void lap_callback(const std_msgs::msg::Int8::SharedPtr msg) 
    {
      lap=msg->data;
    }

    void cones_callback (const eufs_msgs::msg::ConeArray::SharedPtr msg)
    { 
      //Set of cones from SLAM

      RCLCPP_INFO(this->get_logger(), "Recieved cones from SLAM");

      //set of blue and yellow cones from the track
      for (auto e : msg->blue_cones)
      {
        perception_data.bluecones.emplace_back(e.x, e.y);
      }      
      for (auto e : msg->yellow_cones)
      {
        perception_data.yellowcones.emplace_back(e.x, e.y);
      }

    //MIGHT NEED BUT LEFT OUT ORANGE CONES
    //   for (auto e : msg->orange_cones)
    //   {
    //     perception_data.orangecones.emplace_back(e.x, e.y);
    //   }     

      //TODO: shouldn't return a spline

      //this returns a single spline
      Spline midline = generator_mid.spline_from_cones(this->get_logger(), perception_data);

      

      //Add spline here

      interfaces::msg::Spline message = splineToMessage(midline);

      publisher_rcl_pt->publish(message);
      perception_data.bluecones.clear();
      perception_data.yellowcones.clear();
      perception_data.orangecones.clear();
      RCLCPP_INFO(this->get_logger(), "published midpoint spline\n");
      return;
    }


    
    
  public:
    RacelineNode()
    : Node("raceline")
    {
      subscription_cones = this->create_subscription<eufs_msgs::msg::ConeArray>(CONES_TOPIC, 10, std::bind(&RacelineNode::cones_callback, this, _1));
      publisher_rcl_pt = this->create_publisher<interfaces::msg::SplineList>(RACELINE_TOPIC,10);
      generator_mid = MidpointGenerator(10);
    }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<RacelineNode>();
  RCLCPP_INFO(node->get_logger(), "got output\n");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}