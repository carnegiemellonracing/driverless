// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/WheelOdometryErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WHEEL_ODOMETRY_ERR__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__WHEEL_ODOMETRY_ERR__TRAITS_HPP_

#include "eufs_msgs/msg/detail/wheel_odometry_err__struct.hpp"
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
  const eufs_msgs::msg::WheelOdometryErr & msg,
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

  // member: position_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "position_err: ";
    value_to_yaml(msg.position_err, out);
    out << "\n";
  }

  // member: orientation_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "orientation_err: ";
    value_to_yaml(msg.orientation_err, out);
    out << "\n";
  }

  // member: linear_vel_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "linear_vel_err: ";
    value_to_yaml(msg.linear_vel_err, out);
    out << "\n";
  }

  // member: angular_vel_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "angular_vel_err: ";
    value_to_yaml(msg.angular_vel_err, out);
    out << "\n";
  }

  // member: forward_vel_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "forward_vel_err: ";
    value_to_yaml(msg.forward_vel_err, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::WheelOdometryErr & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::WheelOdometryErr>()
{
  return "eufs_msgs::msg::WheelOdometryErr";
}

template<>
inline const char * name<eufs_msgs::msg::WheelOdometryErr>()
{
  return "eufs_msgs/msg/WheelOdometryErr";
}

template<>
struct has_fixed_size<eufs_msgs::msg::WheelOdometryErr>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<eufs_msgs::msg::WheelOdometryErr>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<eufs_msgs::msg::WheelOdometryErr>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__WHEEL_ODOMETRY_ERR__TRAITS_HPP_
