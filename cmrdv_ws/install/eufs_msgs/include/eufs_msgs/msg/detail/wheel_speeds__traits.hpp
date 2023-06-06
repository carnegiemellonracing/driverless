// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/WheelSpeeds.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__TRAITS_HPP_

#include "eufs_msgs/msg/detail/wheel_speeds__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::WheelSpeeds & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: steering
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "steering: ";
    value_to_yaml(msg.steering, out);
    out << "\n";
  }

  // member: lf_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "lf_speed: ";
    value_to_yaml(msg.lf_speed, out);
    out << "\n";
  }

  // member: rf_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rf_speed: ";
    value_to_yaml(msg.rf_speed, out);
    out << "\n";
  }

  // member: lb_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "lb_speed: ";
    value_to_yaml(msg.lb_speed, out);
    out << "\n";
  }

  // member: rb_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rb_speed: ";
    value_to_yaml(msg.rb_speed, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::WheelSpeeds & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::WheelSpeeds>()
{
  return "eufs_msgs::msg::WheelSpeeds";
}

template<>
inline const char * name<eufs_msgs::msg::WheelSpeeds>()
{
  return "eufs_msgs/msg/WheelSpeeds";
}

template<>
struct has_fixed_size<eufs_msgs::msg::WheelSpeeds>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<eufs_msgs::msg::WheelSpeeds>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<eufs_msgs::msg::WheelSpeeds>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__TRAITS_HPP_
