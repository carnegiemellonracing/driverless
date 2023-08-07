// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/PurePursuitCheckpoint.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__TRAITS_HPP_

#include "eufs_msgs/msg/detail/pure_pursuit_checkpoint__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<eufs_msgs::msg::PurePursuitCheckpoint>()
{
  return "eufs_msgs::msg::PurePursuitCheckpoint";
}

template<>
inline const char * name<eufs_msgs::msg::PurePursuitCheckpoint>()
{
  return "eufs_msgs/msg/PurePursuitCheckpoint";
}

template<>
struct has_fixed_size<eufs_msgs::msg::PurePursuitCheckpoint>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Point>::value> {};

template<>
struct has_bounded_size<eufs_msgs::msg::PurePursuitCheckpoint>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Point>::value> {};

template<>
struct is_message<eufs_msgs::msg::PurePursuitCheckpoint>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__TRAITS_HPP_
