// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cmrdv_interfaces:msg/Brakes.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__BRAKES__TRAITS_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__BRAKES__TRAITS_HPP_

#include "cmrdv_interfaces/msg/detail/brakes__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'last_fired'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<cmrdv_interfaces::msg::Brakes>()
{
  return "cmrdv_interfaces::msg::Brakes";
}

template<>
inline const char * name<cmrdv_interfaces::msg::Brakes>()
{
  return "cmrdv_interfaces/msg/Brakes";
}

template<>
struct has_fixed_size<cmrdv_interfaces::msg::Brakes>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct has_bounded_size<cmrdv_interfaces::msg::Brakes>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct is_message<cmrdv_interfaces::msg::Brakes>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CMRDV_INTERFACES__MSG__DETAIL__BRAKES__TRAITS_HPP_
