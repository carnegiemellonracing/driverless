// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/FullState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__FULL_STATE__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__FULL_STATE__TRAITS_HPP_

#include "eufs_msgs/msg/detail/full_state__struct.hpp"
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
  const eufs_msgs::msg::FullState & msg,
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

  // member: x_pos
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x_pos: ";
    value_to_yaml(msg.x_pos, out);
    out << "\n";
  }

  // member: y_pos
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y_pos: ";
    value_to_yaml(msg.y_pos, out);
    out << "\n";
  }

  // member: yaw
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw: ";
    value_to_yaml(msg.yaw, out);
    out << "\n";
  }

  // member: roll
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "roll: ";
    value_to_yaml(msg.roll, out);
    out << "\n";
  }

  // member: u_x
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "u_x: ";
    value_to_yaml(msg.u_x, out);
    out << "\n";
  }

  // member: u_y
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "u_y: ";
    value_to_yaml(msg.u_y, out);
    out << "\n";
  }

  // member: yaw_mder
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "yaw_mder: ";
    value_to_yaml(msg.yaw_mder, out);
    out << "\n";
  }

  // member: front_throttle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "front_throttle: ";
    value_to_yaml(msg.front_throttle, out);
    out << "\n";
  }

  // member: rear_throttle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rear_throttle: ";
    value_to_yaml(msg.rear_throttle, out);
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
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::FullState & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::FullState>()
{
  return "eufs_msgs::msg::FullState";
}

template<>
inline const char * name<eufs_msgs::msg::FullState>()
{
  return "eufs_msgs/msg/FullState";
}

template<>
struct has_fixed_size<eufs_msgs::msg::FullState>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<eufs_msgs::msg::FullState>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<eufs_msgs::msg::FullState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__FULL_STATE__TRAITS_HPP_
