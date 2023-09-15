#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/int8.hpp"
#include "msg/optimizer_points.hpp"
#include "msg/cone_list.hpp"
#include "msg/points.hpp"
#include "generator.hpp"
#include "frenet.hpp"
#include "runpy.hpp"

using std::placeholders::_1;
#define DELTA 0.5
struct raceline_pt{
  double x,y,w_l,w_r;
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

      if((perception_data.yellowcones.size()==0 or perception_data.bluecones.size()==0) && perception_data.orangecones.size()<2){
        return;
      }

      Spline spline = generator_mid.spline_from_cones(perception_data);
      Spline spline_left = generator_left.spline_from_curve(perception_data.bluecones);
      Spline spline_right = generator_right.spline_from_curve(perception_data.yellowcones);


      std::vector<double> rcl_pt_x,rcl_pt_y,rcl_pt_wr, rcl_pt_wl;
      double x,y,wl,wr,rptr,lptr;

      for(int i =0;i<generator_mid.cumulated_splines.size();i++){
        auto e = generator_mid.cumulated_splines[i];

        for(int j=0;j<e.get_points()->size2-1;j++){
          x=gsl_matrix_get(e.get_points(),0,j);
          y=gsl_matrix_get(e.get_points(),1,j);
          double len=0; 
          if (i>0) len = generator_mid.cumulated_lengths[i-1];

          wl = frenet(x,y,generator_left.cumulated_splines,generator_left.cumulated_lengths,generator_mid.cumulated_lengths[i-1]).min_distance;
          wr = frenet(x,y,generator_right.cumulated_splines,generator_right.cumulated_lengths,generator_mid.cumulated_lengths[i-1]).min_distance;

        }
      }


      // auto message  = interfaces::msg::OptimizerPoints();
      // message.x = rcl_pt_x;
      // message.y = rcl_pt_y;
      // message.wl = rcl_pt_wl;
      // message.wr = rcl_pt_wr;
      // publisher_rcl_pt->publish(message);

    }



    perceptionsData perception_data;

    rclcpp::Subscription<interfaces::msg::ConeList>::SharedPtr subscription_cones;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_lap_num;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_rcl_pt;

    int LOOKAHEAD_NEAR = 2;
    int LOOKAHEAD_FAR = 3;
    std::vector<double> vis_lookaheads;
    int lap = 1;
    bool vis_spline = true;
    MidpointGenerator generator_mid;
    MidpointGenerator generator_left;
    MidpointGenerator generator_right;
    
  public:
    MidpointNode()
    : Node("midpoint")
    {
      subscription_cones = this->create_subscription<interfaces::msg::ConeList>("/stereo_cones", 10, std::bind(&MidpointNode::cones_callback, this, _1));
      subscription_lap_num = this->create_subscription<std_msgs::msg::String>("/lap_num", 10, std::bind(&MidpointNode::lap_callback, this, _1));
      publisher_rcl_pt = this->create_publisher<interfaces::msg::OptimizerPoints>("/raceline_points",10);
      //     rclcpp::TimerBase::SharedPtr  timer_ = this->create_wall_timer(
      // 500ms, std::bind(&MinimalPublisher::timer_callback, this));
      generator_mid = MidpointGenerator(10);
      generator_left = MidpointGenerator(10);
      generator_right = MidpointGenerator(10);
      // VIS LOOKAHEADS
    }
};

