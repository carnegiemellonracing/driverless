#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "nodes/controller_component.cpp"


// multi-threaded executor


int main(int argc, char * argv[])
{
  /// Component container with a multi-threaded executor.
  rclcpp::init(argc, argv);
  auto exec = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
  auto node = std::make_shared<ControllerComponent>();
  node->initialize(exec);
  exec->add_node(node->get_node_base_interface());
  exec->spin();
}