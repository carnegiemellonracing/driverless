// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/PlanningMode.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__TRAITS_HPP_

#include "eufs_msgs/msg/detail/planning_mode__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

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
