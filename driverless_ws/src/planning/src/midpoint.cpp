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
#include "generator.hpp"
// #include "frenet.hpp"
// #include "runpy.hpp"


//publish topic example
//ros2 topic pub -1 /stereo_cones eufs_msgs/msg/ConeArray "{blue_cones: [{x: 1.0, y: 2.0, z: 3.0}]}"                                                       


using std::placeholders::_1;
#define DELTA 0.5
struct raceline_pt{
  double x,y,w_l,w_r;
};

class MidpointNode : public rclcpp::Node
{
  private:
    perceptionsData perception_data;

    rclcpp::Subscription<eufs_msgs::msg::ConeArray>::SharedPtr subscription_cones;
    // rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_lap_num;
    rclcpp::Publisher<eufs_msgs::msg::PointArray>::SharedPtr publisher_rcl_pt;

    static const int LOOKAHEAD_NEAR = 2;
    static const int LOOKAHEAD_FAR = 3;
    std::vector<double> vis_lookaheads;
    int lap = 1;
    bool vis_spline = true;
    MidpointGenerator generator_mid;
    MidpointGenerator generator_left;
    MidpointGenerator generator_right;

    void lap_callback(const std_msgs::msg::Int8::SharedPtr msg) 
    {
      lap=msg->data;
    }

    void cones_callback (const eufs_msgs::msg::ConeArray::SharedPtr msg)
    { 
      RCLCPP_INFO(this->get_logger(), "Hello");
      return;
      if (lap>1) return;

      if((msg->blue_cones.size()==0 || msg->yellow_cones.size()==0) && (msg->orange_cones.size()<2)){
        return;
      }


      for (auto e : msg->blue_cones)
      {
        perception_data.bluecones.emplace_back(e.x, e.y);
      }      
      for (auto e : msg->orange_cones)
      {
        perception_data.orangecones.emplace_back(e.x, e.y);
      }      
      for (auto e : msg->yellow_cones)
      {
        perception_data.yellowcones.emplace_back(e.x, e.y);
      }

      //TODO: shouldn't return a spline
      generator_mid.spline_from_cones(perception_data);
      
    
      // Spline spline_left = generator_left.spline_from_curve(perception_data.bluecones);
      // Spline spline_right = generator_right.spline_from_curve(perception_data.yellowcones);


      // WILL BE USED WHEN OPTIMIZER STARTS
      std::vector<double> rcl_pt_x,rcl_pt_y;//,rcl_pt_wr, rcl_pt_wl;
      double x,y;//,wl,wr,rptr,lptr;
      eufs_msgs::msg::PointArray message  = eufs_msgs::msg::PointArray();
      std::vector<geometry_msgs::msg::Point> Points;

      //
      for(unsigned int i =0;i<generator_mid.cumulated_splines.size();i++){
        auto spline = generator_mid.cumulated_splines[i];
        //TODO:create a typedef, but size2 is the num of rows
        for(unsigned int j=0;j<spline.get_points()->size2-1;j++){
          x=gsl_matrix_get(spline.get_points(),0,j);
          y=gsl_matrix_get(spline.get_points(),1,j);
          // double len=0; 
          // if (i>0) len = generator_mid.cumulated_lengths[i-1];
          geometry_msgs::msg::Point tmpPoint;
          tmpPoint.x=x;
          tmpPoint.y=y;
          Points.push_back(tmpPoint);
          // wl = frenet(x,y,generator_left.cumulated_splines,generator_left.cumulated_lengths,generator_mid.cumulated_lengths[i-1]).min_distance;
          // wr = frenet(x,y,generator_right.cumulated_splines,generator_right.cumulated_lengths,generator_mid.cumulated_lengths[i-1]).min_distance;
        }
      }
      message.set__points(Points);
      publisher_rcl_pt->publish(message);
    }


    
  public:
    MidpointNode()
    : Node("midpoint")
    {
      subscription_cones = this->create_subscription<eufs_msgs::msg::ConeArray>("/stereo_cones", 10, std::bind(&MidpointNode::cones_callback, this, _1));
      // subscription_lap_num = this->create_subscription<std_msgs::msg::String>("/lap_num", 10, std::bind(&MidpointNode::lap_callback, this, _1));
      publisher_rcl_pt = this->create_publisher<eufs_msgs::msg::PointArray>("/midpoint_points",10);
      // publisher_rcl_pt = this->create_publisher<std_msgs::msg::String>("/midpoint_points",10);
      //     rclcpp::TimerBase::SharedPtr  timer_ = this->create_wall_timer(
      // 500ms, std::bind(&MinimalPublisher::timer_callback, this));
      generator_mid = MidpointGenerator(10);
      generator_left = MidpointGenerator(10);
      generator_right = MidpointGenerator(10);
      // VIS LOOKAHEADS
    }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MidpointNode>());
  rclcpp::shutdown();
  return 0;
}
