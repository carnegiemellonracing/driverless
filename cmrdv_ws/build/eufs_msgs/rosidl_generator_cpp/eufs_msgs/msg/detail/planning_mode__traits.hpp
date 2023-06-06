// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/PlanningMode.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__TRAITS_HPP_

#include "eufs_msgs/msg/detail/planning_mode__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::PlanningMode & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: mode
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mode: ";
    value_to_yaml(msg.mode, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::PlanningMode & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::PlanningMode>()
{
  return "eufs_msgs::msg::PlanningMode";
}

template<>
inline const char * name<eufs_msgs::msg::PlanningMode>()
{
  return "eufs_msgs/msg/PlanningMode";
}

template<>
struct has_fixed_size<eufs_msgs::msg::PlanningMode>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<eufs_msgs::msg::PlanningMode>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<eufs_msgs::msg::PlanningMode>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__TRAITS_HPP_
