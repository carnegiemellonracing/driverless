#pragma once

#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <functional>
#include <memory>
#include <cstdint>

/**
 * A simple ROS2 node that takes in 4 command line arguments and calls sendControlAction on a 10Hz loop.
 * Arguments: front_torque_mNm, back_torque_mNm, velocity_rpm, rack_displacement_adc
 */
class ControlSenderNode : public rclcpp::Node
{
public:
  /**
   * Constructor for the ControlSenderNode
   * 
   * @param front_torque_mNm Front torque in milli-Newton-meters
   * @param back_torque_mNm Back torque in milli-Newton-meters
   * @param velocity_rpm Maximum velocity in RPM
   * @param rack_displacement_adc Rack displacement in ADC units
   */
  ControlSenderNode(int16_t front_torque_mNm, int16_t back_torque_mNm, uint16_t velocity_rpm, uint16_t rack_displacement_adc);

private:
  /**
   * Timer callback that sends the control action
   */
  void timer_callback();

  // Store the command line arguments
  int16_t front_torque_mNm_;
  int16_t back_torque_mNm_;
  uint16_t velocity_rpm_;
  uint16_t rack_displacement_adc_;
  
  // Timer for periodic execution
  rclcpp::TimerBase::SharedPtr timer_;
}; 