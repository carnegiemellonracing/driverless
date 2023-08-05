// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from interfaces:msg/ConePositions.idl
// generated code does not contain a copyright notice

#ifndef INTERFACES__MSG__DETAIL__CONE_POSITIONS__TRAITS_HPP_
#define INTERFACES__MSG__DETAIL__CONE_POSITIONS__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "interfaces/msg/detail/cone_positions__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'cone_positions'
#include "std_msgs/msg/detail/float32__traits.hpp"

namespace interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const ConePositions & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: cone_positions
  {
    if (msg.cone_positions.size() == 0) {
      out << "cone_positions: []";
    } else {
      out << "cone_positions: [";
      size_t pending_items = msg.cone_positions.size();
      for (auto item : msg.cone_positions) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const ConePositions & msg,
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

  // member: cone_positions
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.cone_positions.size() == 0) {
      out << "cone_positions: []\n";
    } else {
      out << "cone_positions:\n";
      for (auto item : msg.cone_positions) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ConePositions & msg, bool use_flow_style = false)
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
  const interfaces::msg::ConePositions & msg,
  std::ostream & out, size_t indentation = 0)
{
  interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const interfaces::msg::ConePositions & msg)
{
  return interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<interfaces::msg::ConePositions>()
{
  return "interfaces::msg::ConePositions";
}

template<>
inline const char * name<interfaces::msg::ConePositions>()
{
  return "interfaces/msg/ConePositions";
}

template<>
struct has_fixed_size<interfaces::msg::ConePositions>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<interfaces::msg::ConePositions>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<interfaces::msg::ConePositions>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // INTERFACES__MSG__DETAIL__CONE_POSITIONS__TRAITS_HPP_
