// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/SLAMState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__SLAM_STATE__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__SLAM_STATE__TRAITS_HPP_

#include "eufs_msgs/msg/detail/slam_state__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::SLAMState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: loop_closed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "loop_closed: ";
    value_to_yaml(msg.loop_closed, out);
    out << "\n";
  }

  // member: laps
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "laps: ";
    value_to_yaml(msg.laps, out);
    out << "\n";
  }

  // member: status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status: ";
    value_to_yaml(msg.status, out);
    out << "\n";
  }

  // member: state
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "state: ";
    value_to_yaml(msg.state, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::SLAMState & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::SLAMState>()
{
  return "eufs_msgs::msg::SLAMState";
}

template<>
inline const char * name<eufs_msgs::msg::SLAMState>()
{
  return "eufs_msgs/msg/SLAMState";
}

template<>
struct has_fixed_size<eufs_msgs::msg::SLAMState>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<eufs_msgs::msg::SLAMState>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<eufs_msgs::msg::SLAMState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__SLAM_STATE__TRAITS_HPP_
