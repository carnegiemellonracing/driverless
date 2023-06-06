// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/ChassisState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CHASSIS_STATE__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__CHASSIS_STATE__TRAITS_HPP_

#include "eufs_msgs/msg/detail/chassis_state__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::ChassisState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_yaml(msg.header, out, indentation + 2);
  }

  // member: throttle_relay_enabled
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "throttle_relay_enabled: ";
    value_to_yaml(msg.throttle_relay_enabled, out);
    out << "\n";
  }

  // member: autonomous_enabled
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "autonomous_enabled: ";
    value_to_yaml(msg.autonomous_enabled, out);
    out << "\n";
  }

  // member: runstop_motion_enabled
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "runstop_motion_enabled: ";
    value_to_yaml(msg.runstop_motion_enabled, out);
    out << "\n";
  }

  // member: steering_commander
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "steering_commander: ";
    value_to_yaml(msg.steering_commander, out);
    out << "\n";
  }

  // member: steering
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "steering: ";
    value_to_yaml(msg.steering, out);
    out << "\n";
  }

  // member: throttle_commander
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "throttle_commander: ";
    value_to_yaml(msg.throttle_commander, out);
    out << "\n";
  }

  // member: throttle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "throttle: ";
    value_to_yaml(msg.throttle, out);
    out << "\n";
  }

  // member: front_brake_commander
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "front_brake_commander: ";
    value_to_yaml(msg.front_brake_commander, out);
    out << "\n";
  }

  // member: front_brake
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "front_brake: ";
    value_to_yaml(msg.front_brake, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::ChassisState & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::ChassisState>()
{
  return "eufs_msgs::msg::ChassisState";
}

template<>
inline const char * name<eufs_msgs::msg::ChassisState>()
{
  return "eufs_msgs/msg/ChassisState";
}

template<>
struct has_fixed_size<eufs_msgs::msg::ChassisState>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<eufs_msgs::msg::ChassisState>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<eufs_msgs::msg::ChassisState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__CHASSIS_STATE__TRAITS_HPP_
