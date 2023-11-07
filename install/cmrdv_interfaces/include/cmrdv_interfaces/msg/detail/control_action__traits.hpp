// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cmrdv_interfaces:msg/ControlAction.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__TRAITS_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__TRAITS_HPP_

#include "cmrdv_interfaces/msg/detail/control_action__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<cmrdv_interfaces::msg::ControlAction>()
{
  return "cmrdv_interfaces::msg::ControlAction";
}

template<>
inline const char * name<cmrdv_interfaces::msg::ControlAction>()
{
  return "cmrdv_interfaces/msg/ControlAction";
}

template<>
struct has_fixed_size<cmrdv_interfaces::msg::ControlAction>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<cmrdv_interfaces::msg::ControlAction>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<cmrdv_interfaces::msg::ControlAction>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__TRAITS_HPP_
