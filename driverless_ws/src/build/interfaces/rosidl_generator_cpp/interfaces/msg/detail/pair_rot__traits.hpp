// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from interfaces:msg/PairROT.idl
// generated code does not contain a copyright notice

#ifndef INTERFACES__MSG__DETAIL__PAIR_ROT__TRAITS_HPP_
#define INTERFACES__MSG__DETAIL__PAIR_ROT__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "interfaces/msg/detail/pair_rot__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'near'
// Member 'far'
#include "interfaces/msg/detail/car_rot__traits.hpp"

namespace interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const PairROT & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: near
  {
    out << "near: ";
    to_flow_style_yaml(msg.near, out);
    out << ", ";
  }

  // member: far
  {
    out << "far: ";
    to_flow_style_yaml(msg.far, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PairROT & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: near
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "near:\n";
    to_block_style_yaml(msg.near, out, indentation + 2);
  }

  // member: far
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "far:\n";
    to_block_style_yaml(msg.far, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PairROT & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace interfaces

namespace rosidl_generator_traits
{

[[deprecated("use interfaces::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const interfaces::msg::PairROT & msg,
  std::ostream & out, size_t indentation = 0)
{
  interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const interfaces::msg::PairROT & msg)
{
  return interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<interfaces::msg::PairROT>()
{
  return "interfaces::msg::PairROT";
}

template<>
inline const char * name<interfaces::msg::PairROT>()
{
  return "interfaces/msg/PairROT";
}

template<>
struct has_fixed_size<interfaces::msg::PairROT>
  : std::integral_constant<bool, has_fixed_size<interfaces::msg::CarROT>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<interfaces::msg::PairROT>
  : std::integral_constant<bool, has_bounded_size<interfaces::msg::CarROT>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<interfaces::msg::PairROT>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // INTERFACES__MSG__DETAIL__PAIR_ROT__TRAITS_HPP_
