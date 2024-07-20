#include <memory>

#include "rclcpp/rclcpp.hpp"

#include "nodes/controller_component.cpp"


// single-threaded executor
int main(int argc, char * argv[])
{
  /// Component container with a single-threaded executor.
  rclcpp::init(argc, argv);
  auto exec = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  auto node = std::make_shared<controller::ControllerComponent>();
  // node->initialize(exec);
  exec->add_node(node->get_node_base_interface());
  exec->spin();
}