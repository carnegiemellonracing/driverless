// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cmrdv_interfaces:msg/ConePositions.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__TRAITS_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__TRAITS_HPP_

#include "cmrdv_interfaces/msg/detail/cone_positions__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<cmrdv_interfaces::msg::ConePositions>()
{
  return "cmrdv_interfaces::msg::ConePositions";
}

template<>
inline const char * name<cmrdv_interfaces::msg::ConePositions>()
{
  return "cmrdv_interfaces/msg/ConePositions";
}

template<>
struct has_fixed_size<cmrdv_interfaces::msg::ConePositions>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cmrdv_interfaces::msg::ConePositions>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cmrdv_interfaces::msg::ConePositions>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__TRAITS_HPP_
