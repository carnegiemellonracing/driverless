// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cmrdv_interfaces:msg/CarROT.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__TRAITS_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__TRAITS_HPP_

#include "cmrdv_interfaces/msg/detail/car_rot__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<cmrdv_interfaces::msg::CarROT>()
{
  return "cmrdv_interfaces::msg::CarROT";
}

template<>
inline const char * name<cmrdv_interfaces::msg::CarROT>()
{
  return "cmrdv_interfaces/msg/CarROT";
}

template<>
struct has_fixed_size<cmrdv_interfaces::msg::CarROT>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<cmrdv_interfaces::msg::CarROT>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<cmrdv_interfaces::msg::CarROT>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__TRAITS_HPP_