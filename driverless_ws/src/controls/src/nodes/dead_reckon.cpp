#include "dead_reckon.hpp"
#include <can/cmr_can.h> // For sendControlAction
#include <string>
#include <cmath>

// External declaration for the swangle_to_adc function
extern "C" uint16_t swangle_to_adc(float swangle);

using namespace std::chrono_literals;

/**
 * A ROS2 node that sends control actions to the vehicle.
 * It supports two modes:
 * 1. Straight mode: Constant torque and steering angle
 * 2. Sinusoidal mode: Constant torque with oscillating steering angle
 */
ControlSenderNode::ControlSenderNode(int16_t front_torque_mNm, int16_t back_torque_mNm, uint16_t velocity_rpm, float angle_degrees)
: Node("control_sender"), 
  front_torque_mNm_(front_torque_mNm),
  back_torque_mNm_(back_torque_mNm),
  velocity_rpm_(velocity_rpm),
  angle_degrees_(angle_degrees),
  sinusoid_mode_(false)
{
  // Convert angle from degrees to radians, then to ADC value
  float angle_rad = angle_degrees * M_PI / 180.0f;
  rack_displacement_adc_ = swangle_to_adc(angle_rad);
  
  // Create a timer that triggers every 100ms (10Hz)
  timer_ = this->create_wall_timer(
    100ms, std::bind(&ControlSenderNode::timer_callback, this));
  
  // Convert from mNm to Nm for display
  float front_torque_Nm = front_torque_mNm_ / 1000.0f;
  float back_torque_Nm = back_torque_mNm_ / 1000.0f;
  
  RCLCPP_INFO(this->get_logger(), "Starting control sender in STRAIGHT mode with parameters:");
  RCLCPP_INFO(this->get_logger(), "  front_torque_Nm: %.3f", front_torque_Nm);
  RCLCPP_INFO(this->get_logger(), "  back_torque_Nm: %.3f", back_torque_Nm);
  RCLCPP_INFO(this->get_logger(), "  velocity_rpm: %u", velocity_rpm_);
  RCLCPP_INFO(this->get_logger(), "  angle_degrees: %.2f (ADC: %u)", angle_degrees_, rack_displacement_adc_);
}

ControlSenderNode::ControlSenderNode(int16_t torque_mNm, uint16_t velocity_rpm, float angle_degrees, float period_seconds)
: Node("control_sender"),
  front_torque_mNm_(torque_mNm),
  back_torque_mNm_(torque_mNm),
  velocity_rpm_(velocity_rpm),
  rack_displacement_adc_(0), // Will be calculated in timer callback
  sinusoid_mode_(true),
  angle_degrees_(angle_degrees),
  period_seconds_(period_seconds),
  start_time_(this->now())
{
  // Create a timer that triggers every 100ms (10Hz)
  timer_ = this->create_wall_timer(
    100ms, std::bind(&ControlSenderNode::timer_callback, this));
  
  // Convert from mNm to Nm for display
  float torque_Nm = torque_mNm / 1000.0f;
  
  RCLCPP_INFO(this->get_logger(), "Starting control sender in SINUSOIDAL mode with parameters:");
  RCLCPP_INFO(this->get_logger(), "  torque_Nm: %.3f", torque_Nm);
  RCLCPP_INFO(this->get_logger(), "  velocity_rpm: %u", velocity_rpm_);
  RCLCPP_INFO(this->get_logger(), "  angle_degrees: %.2f", angle_degrees_);
  RCLCPP_INFO(this->get_logger(), "  period_seconds: %.2f", period_seconds_);
}

float ControlSenderNode::calculate_steering_angle() {
  // Calculate elapsed time since start
  auto elapsed = this->now() - start_time_;
  double elapsed_seconds = elapsed.seconds();
  
  // Calculate the current position in the sinusoidal cycle
  double cycle_position = fmod(elapsed_seconds, period_seconds_) / period_seconds_; // Normalized to [0, 1]
  
  // Convert angle from degrees to radians
  float max_angle_rad = angle_degrees_ * M_PI / 180.0f;
  
  // Calculate the sinusoidal value between -max_angle_rad and +max_angle_rad
  float current_angle = max_angle_rad * sin(2.0f * M_PI * cycle_position);
  
  RCLCPP_DEBUG(this->get_logger(), "Current steering angle: %f radians (%f degrees)", 
               current_angle, current_angle * 180.0f / M_PI);
  return current_angle;
}

void ControlSenderNode::timer_callback()
{
  // For sinusoidal mode, calculate the current steering angle
  if (sinusoid_mode_) {
    float current_angle = calculate_steering_angle();
    // Convert to ADC value
    rack_displacement_adc_ = swangle_to_adc(current_angle);
  }
  
  // Call sendControlAction with the provided arguments
  int result = sendControlAction(
    front_torque_mNm_,
    back_torque_mNm_,
    velocity_rpm_,
    rack_displacement_adc_
  );
  
  // Convert from mNm to Nm for display
  float front_torque_Nm = front_torque_mNm_ / 1000.0f;
  float back_torque_Nm = back_torque_mNm_ / 1000.0f;
  
  RCLCPP_INFO(this->get_logger(), "Sending control action with parameters: front_torque_Nm: %.3f, back_torque_Nm: %.3f, velocity_rpm: %u, rack_displacement_adc: %u", 
              front_torque_Nm, back_torque_Nm, velocity_rpm_, rack_displacement_adc_);
  
  if (result != 0) {
    RCLCPP_ERROR(this->get_logger(), "Failed to send control action, error code: %d", result);
  } else {
    if (sinusoid_mode_) {
      RCLCPP_DEBUG(this->get_logger(), "Sent sinusoidal control action, angle: %f rad, ADC: %u", 
                  calculate_steering_angle(), rack_displacement_adc_);
    } else {
      RCLCPP_DEBUG(this->get_logger(), "Control action sent successfully");
    }
  }
}

void print_usage(const char* prog_name) {
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  %s --straight <front_torque_Nm> <back_torque_Nm> <velocity_rpm> <angle_degrees>\n", prog_name);
  fprintf(stderr, "  %s --sinusoid <torque_Nm> <velocity_rpm> <angle_degrees> <period_seconds>\n", prog_name);
  fprintf(stderr, "  %s --help\n", prog_name);
}

void print_help() {
  printf("\nControl Sender - A utility to send control actions to the vehicle\n");
  printf("\nMODES:\n");
  
  printf("\n1. STRAIGHT MODE (--straight):\n");
  printf("   Sends constant control values to the vehicle.\n");
  printf("   Parameters:\n");
  printf("   - front_torque_Nm: Front torque in Newton-meters [float]\n");
  printf("   - back_torque_Nm: Back torque in Newton-meters [float]\n");
  printf("   - velocity_rpm: Maximum velocity in RPM [uint16]\n");
  printf("   - angle_degrees: Steering wheel angle in degrees [float]\n");
  printf("   Example: --straight 0.1 0.1 3000 10.5\n");
  
  printf("\n2. SINUSOIDAL MODE (--sinusoid):\n");
  printf("   Sends a constant torque but varies the steering angle in a sinusoidal pattern.\n");
  printf("   Parameters:\n");
  printf("   - torque_Nm: Constant torque for both front and back in Newton-meters [float]\n");
  printf("   - velocity_rpm: Maximum velocity in RPM [uint16]\n");
  printf("   - angle_degrees: Maximum steering angle in degrees (oscillates between -angle and +angle) [float]\n");
  printf("   - period_seconds: Period of the sinusoidal oscillation in seconds [float]\n");
  printf("   Example: --sinusoid 0.1 3000 10.0 2.5\n");
  
  printf("\nHELP OPTION (--help):\n");
  printf("   Displays this help message.\n\n");
}

int main(int argc, char * argv[])
{
  // Initialize ROS
  rclcpp::init(argc, argv);
  
  // Need at least one argument for mode
  if (argc < 2) {
    RCLCPP_ERROR(rclcpp::get_logger("control_sender"), "Missing mode argument (--straight, --sinusoid, or --help)");
    print_usage(argv[0]);
    return 1;
  }
  
  std::string mode = argv[1];
  
  // Handle help option
  if (mode == "--help") {
    print_help();
    return 0;
  }
  
  std::shared_ptr<ControlSenderNode> node;
  
  if (mode == "--straight") {
    // Check if we have the correct number of arguments
    if (argc != 6) {
      RCLCPP_ERROR(rclcpp::get_logger("control_sender"), 
                  "Straight mode requires 4 arguments: <front_torque_Nm> <back_torque_Nm> <velocity_rpm> <angle_degrees>");
      print_usage(argv[0]);
      return 1;
    }
    
    // Parse command line arguments and convert Nm to mNm
    float front_torque_Nm = std::stof(argv[2]);
    float back_torque_Nm = std::stof(argv[3]);
    int16_t front_torque_mNm = static_cast<int16_t>(front_torque_Nm * 1000.0f);
    int16_t back_torque_mNm = static_cast<int16_t>(back_torque_Nm * 1000.0f);
    uint16_t velocity_rpm = static_cast<uint16_t>(std::stoi(argv[4]));
    float angle_degrees = std::stof(argv[5]);
    
    // Create and run the node in straight mode
    node = std::make_shared<ControlSenderNode>(
      front_torque_mNm,
      back_torque_mNm,
      velocity_rpm,
      angle_degrees
    );
  } else if (mode == "--sinusoid") {
    // Check if we have the correct number of arguments
    if (argc != 6) {
      RCLCPP_ERROR(rclcpp::get_logger("control_sender"), 
                  "Sinusoid mode requires 4 arguments: <torque_Nm> <velocity_rpm> <angle_degrees> <period_seconds>");
      print_usage(argv[0]);
      return 1;
    }
    
    // Parse command line arguments and convert Nm to mNm
    float torque_Nm = std::stof(argv[2]);
    int16_t torque_mNm = static_cast<int16_t>(torque_Nm * 1000.0f);
    uint16_t velocity_rpm = static_cast<uint16_t>(std::stoi(argv[3]));
    float angle_degrees = std::stof(argv[4]);
    float period_seconds = std::stof(argv[5]);
    
    // Create and run the node in sinusoidal mode
    node = std::make_shared<ControlSenderNode>(
      torque_mNm,
      velocity_rpm,
      angle_degrees,
      period_seconds
    );
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("control_sender"), "Invalid mode: %s. Must be --straight, --sinusoid, or --help", mode.c_str());
    print_usage(argv[0]);
    return 1;
  }
  
  RCLCPP_INFO(rclcpp::get_logger("control_sender"), "Control sender node started");
  
  // Spin the node
  rclcpp::spin(node);
  
  // Clean up
  rclcpp::shutdown();
  return 0;
} 
