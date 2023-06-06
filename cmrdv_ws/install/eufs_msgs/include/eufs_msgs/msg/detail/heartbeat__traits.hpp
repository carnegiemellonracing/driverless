// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/Heartbeat.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__HEARTBEAT__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__HEARTBEAT__TRAITS_HPP_

#include "eufs_msgs/msg/detail/heartbeat__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::Heartbeat & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    value_to_yaml(msg.id, out);
    out << "\n";
  }

  // member: data
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "data: ";
    value_to_yaml(msg.data, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::Heartbeat & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::Heartbeat>()
{
  return "eufs_msgs::msg::Heartbeat";
}

template<>
inline const char * name<eufs_msgs::msg::Heartbeat>()
{
  return "eufs_msgs/msg/Heartbeat";
}

template<>
struct has_fixed_size<eufs_msgs::msg::Heartbeat>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<eufs_msgs::msg::Heartbeat>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<eufs_msgs::msg::Heartbeat>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__HEARTBEAT__TRAITS_HPP_
