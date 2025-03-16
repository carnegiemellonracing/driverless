#include "dead_reckon.hpp"
#include <can/cmr_can.h> // For sendControlAction
#include <string>

using namespace std::chrono_literals;

/**
 * A simple ROS2 node that takes in 4 command line arguments and calls sendControlAction on a 10Hz loop.
 * Arguments: front_torque_mNm, back_torque_mNm, velocity_rpm, rack_displacement_adc
 */
ControlSenderNode::ControlSenderNode(int16_t front_torque_mNm, int16_t back_torque_mNm, uint16_t velocity_rpm, uint16_t rack_displacement_adc)
: Node("control_sender"), 
  front_torque_mNm_(front_torque_mNm),
  back_torque_mNm_(back_torque_mNm),
  velocity_rpm_(velocity_rpm),
  rack_displacement_adc_(rack_displacement_adc)
{
  // Create a timer that triggers every 100ms (10Hz)
  timer_ = this->create_wall_timer(
    100ms, std::bind(&ControlSenderNode::timer_callback, this));
  
  RCLCPP_INFO(this->get_logger(), "Starting control sender with parameters:");
  RCLCPP_INFO(this->get_logger(), "  front_torque_mNm: %d", front_torque_mNm_);
  RCLCPP_INFO(this->get_logger(), "  back_torque_mNm: %d", back_torque_mNm_);
  RCLCPP_INFO(this->get_logger(), "  velocity_rpm: %u", velocity_rpm_);
  RCLCPP_INFO(this->get_logger(), "  rack_displacement_adc: %u", rack_displacement_adc_);
}

void ControlSenderNode::timer_callback()
{
  // Call sendControlAction with the provided arguments
  int result = sendControlAction(
    front_torque_mNm_,
    back_torque_mNm_,
    velocity_rpm_,
    rack_displacement_adc_
  );
  std::cout << "Result: " << result << std::endl;
  
  if (result != 0) {
    RCLCPP_ERROR(this->get_logger(), "Failed to send control action, error code: %d", result);
  } else {
    RCLCPP_DEBUG(this->get_logger(), "Control action sent successfully");
  }
}

int main(int argc, char * argv[])
{
  // Initialize ROS
  rclcpp::init(argc, argv);
  
  // Check if we have the correct number of arguments
  if (argc != 5) {
    RCLCPP_ERROR(rclcpp::get_logger("control_sender"), 
                "Usage: %s <front_torque_mNm> <back_torque_mNm> <velocity_rpm> <rack_displacement_adc>", 
                argv[0]);
    return 1;
  }
  
  // Parse command line arguments
  int16_t front_torque_mNm = static_cast<int16_t>(std::stoi(argv[1]));
  int16_t back_torque_mNm = static_cast<int16_t>(std::stoi(argv[2]));
  uint16_t velocity_rpm = static_cast<uint16_t>(std::stoi(argv[3]));
  uint16_t rack_displacement_adc = static_cast<uint16_t>(std::stoi(argv[4]));
  
  // Create and run the node
  auto node = std::make_shared<ControlSenderNode>(
    front_torque_mNm,
    back_torque_mNm,
    velocity_rpm,
    rack_displacement_adc
  );
  
  RCLCPP_INFO(rclcpp::get_logger("control_sender"), "Control sender node started");
  
  // Spin the node
  rclcpp::spin(node);
  
  // Clean up
  rclcpp::shutdown();
  return 0;
} 
