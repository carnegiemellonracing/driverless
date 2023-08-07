// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/ConeArrayWithCovariance.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__TRAITS_HPP_

#include "eufs_msgs/msg/detail/cone_array_with_covariance__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<eufs_msgs::msg::ConeArrayWithCovariance>()
{
  return "eufs_msgs::msg::ConeArrayWithCovariance";
}

template<>
inline const char * name<eufs_msgs::msg::ConeArrayWithCovariance>()
{
  return "eufs_msgs/msg/ConeArrayWithCovariance";
}

template<>
struct has_fixed_size<eufs_msgs::msg::ConeArrayWithCovariance>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<eufs_msgs::msg::ConeArrayWithCovariance>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<eufs_msgs::msg::ConeArrayWithCovariance>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__TRAITS_HPP_
