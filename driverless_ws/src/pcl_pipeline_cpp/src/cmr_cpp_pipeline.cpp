#include <cstdio>

// ROS2 Imports
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"


class CMRCppPipelineNode : public rclcpp::Node {

  public:
    CMRCppPipelineNode() : Node("cmr_cpp_pipeline_node") {
      
    }

  private:


};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CMRCppPipelineNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
