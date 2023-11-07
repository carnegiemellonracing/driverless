// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cmrdv_interfaces:msg/ConeList.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CONE_LIST__TRAITS_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__CONE_LIST__TRAITS_HPP_

#include "cmrdv_interfaces/msg/detail/cone_list__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<cmrdv_interfaces::msg::ConeList>()
{
  return "cmrdv_interfaces::msg::ConeList";
}

template<>
inline const char * name<cmrdv_interfaces::msg::ConeList>()
{
  return "cmrdv_interfaces/msg/ConeList";
}

template<>
struct has_fixed_size<cmrdv_interfaces::msg::ConeList>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<cmrdv_interfaces::msg::ConeList>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<cmrdv_interfaces::msg::ConeList>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CONE_LIST__TRAITS_HPP_
