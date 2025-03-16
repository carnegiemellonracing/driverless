#pragma once

#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <functional>
#include <memory>
#include <cstdint>
#include <cmath>

/**
 * A ROS2 node that sends control actions to the vehicle.
 * It supports two modes:
 * 1. Straight mode: Sends constant values for all parameters
 * 2. Sinusoidal mode: Sends a sinusoidal steering angle while maintaining constant torque
 */
class ControlSenderNode : public rclcpp::Node
{
public:
  /**
   * Constructor for the ControlSenderNode in straight mode
   * 
   * @param front_torque_mNm Front torque in milli-Newton-meters
   * @param back_torque_mNm Back torque in milli-Newton-meters
   * @param velocity_rpm Maximum velocity in RPM
   * @param angle_degrees Steering wheel angle in degrees
   */
  ControlSenderNode(int16_t front_torque_mNm, int16_t back_torque_mNm, uint16_t velocity_rpm, float angle_degrees);

  /**
   * Constructor for the ControlSenderNode in sinusoidal mode
   * 
   * @param torque_mNm Constant torque for both front and back in milli-Newton-meters
   * @param velocity_rpm Maximum velocity in RPM
   * @param angle_degrees Maximum steering angle in degrees (will oscillate between -angle and +angle)
   * @param period_seconds Period of the sinusoidal oscillation in seconds
   */
  ControlSenderNode(int16_t torque_mNm, uint16_t velocity_rpm, float angle_degrees, float period_seconds);

private:
  /**
   * Timer callback that sends the control action
   */
  void timer_callback();

  /**
   * Calculate the current steering angle for sinusoidal mode
   * 
   * @return Current steering angle in radians
   */
  float calculate_steering_angle();

  // Store the command line arguments for straight mode
  int16_t front_torque_mNm_;
  int16_t back_torque_mNm_;
  uint16_t velocity_rpm_;
  uint16_t rack_displacement_adc_;
  
  // Additional parameters for sinusoidal mode
  bool sinusoid_mode_ = false;
  float angle_degrees_ = 0.0f;
  float period_seconds_ = 1.0f;
  rclcpp::Time start_time_;
  
  // Timer for periodic execution
  rclcpp::TimerBase::SharedPtr timer_;
}; 