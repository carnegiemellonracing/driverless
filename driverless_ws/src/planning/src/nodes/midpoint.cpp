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


//publish topic example
//ros2 topic pub -1 /stereo_node_cones eufs_msgs/msg/ConeArray "{blue_cones: [{x: 1.0, y: 2.0, z: 3.0}]}"   


// ros2 topic pub -1 /stereo_node_cones eufs_msgs/msg/ConeArray "{blue_cones: [{x: 0.0, y: 3.0, z: 0.0}, {x: 1.414, y: 2.236 , z: 0.0}, {x: 3.0, y: 0.0 , z: 0.0}], yellow_cones: [{x: 0.0, y: 2.0, z: 0.0}, {x: 1.414, y: 1.414, z: 0.0}, {x: 2.0, y: 0.0, z: 0.0}]}"   


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
    // rclcpp::Publisher<eufs_msgs::msg::PointArray>::SharedPtr publisher_rcl_pt;
    rclcpp::Publisher<interfaces::msg::Spline>::SharedPtr publisher_rcl_pt;

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
      // return;
      if (lap>1) return;

      // if((msg->blue_cones.size() < 3 || msg->yellow_cones.size() < 3) && (msg->orange_cones.size()<2)){
      if((msg->blue_cones.size() < 3 || msg->yellow_cones.size() < 3)){
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

      Spline midline = generator_mid.spline_from_cones(this->get_logger(), perception_data);
      interfaces::msg::Spline message  = splineToMessage(midline);
      publisher_rcl_pt->publish(message);
      perception_data.bluecones.clear();
      perception_data.yellowcones.clear();
      perception_data.orangecones.clear();
      RCLCPP_INFO(this->get_logger(), "published midpoint spline\n");
      return;
    }


    interfaces::msg::Spline splineToMessage(Spline spline){
        interfaces::msg::Spline message;

        std::fill(std::begin(message.spl_poly_coefs), std::end(message.spl_poly_coefs), 0.0);
        for (int i = 0; i < 4; i ++) {
        message.spl_poly_coefs[i] = spline.spl_poly.nums[i];
        }

        std::fill(std::begin(message.first_der_coefs), std::end(message.first_der_coefs), 0.0);
        for (int i = 0; i < 3; i ++) {
        message.first_der_coefs[i] = spline.first_der.nums[i];
        }

        // message.second_der_coefs.clear();
        std::fill(std::begin(message.second_der_coefs), std::end(message.second_der_coefs), 0.0);
        for (int i = 0; i < 2; i ++) {
        message.second_der_coefs[i] = spline.second_der.nums[i];
        }

        message.points.clear();
        message.rotated_points.clear();

        for (int i = 0; i < spline.points.cols(); i++) {
        geometry_msgs::msg::Point point; // z field is unused
        point.x = spline.points(0, i);
        point.y = spline.points(1, i);
        message.points.push_back(point);

        // geometry_msgs::msg::Point point; // z field is unused
        point.x = spline.rotated_points(0, i);
        point.y = spline.rotated_points(1, i);
        message.rotated_points.push_back(point);
        }

        std::fill(std::begin(message.q), std::end(message.q), 0.0);
        message.q[0] = (spline.Q(0, 0));
        message.q[1] = (spline.Q(0, 1));
        message.q[2] = (spline.Q(1, 0));
        message.q[3] = (spline.Q(1, 1));

        // message.translation_vector.clear();
        std::fill(std::begin(message.translation_vector), std::end(message.translation_vector), 0.0);
        message.translation_vector[0] = spline.translation_vector[0];
        message.translation_vector[1] = spline.translation_vector[1];
        message.path_id = spline.path_id;
        message.sort_index = spline.sort_index;
        message.length = spline.calculateLength();

        return message;
    }



    
  public:
    MidpointNode()
    : Node("midpoint")
    {
      subscription_cones = this->create_subscription<eufs_msgs::msg::ConeArray>("/stereo_node_cones", 10, std::bind(&MidpointNode::cones_callback, this, _1));
      // subscription_lap_num = this->create_subscription<std_msgs::msg::String>("/lap_num", 10, std::bind(&MidpointNode::lap_callback, this, _1));
      publisher_rcl_pt = this->create_publisher<interfaces::msg::Spline>("/midpoint_spline",10);
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
  // // RCLCPP_INFO(this->get_logger(), "Started Midpoint Node");
  // rclcpp::init(argc, argv);
  // rclcpp::spin(std::make_shared<MidpointNode>());
  // rclcpp::shutdown();
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MidpointNode>();
  RCLCPP_INFO(node->get_logger(), "got output\n");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}