// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from interfaces:msg/ConeList.idl
// generated code does not contain a copyright notice

#ifndef INTERFACES__MSG__DETAIL__CONE_LIST__TRAITS_HPP_
#define INTERFACES__MSG__DETAIL__CONE_LIST__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "interfaces/msg/detail/cone_list__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'blue_cones'
// Member 'yellow_cones'
// Member 'orange_cones'
#include "geometry_msgs/msg/detail/point__traits.hpp"

namespace interfaces
{

namespace msg
{

inline void to_flow_style_yaml(
  const ConeList & msg,
  std::ostream & out)
{
  out << "{";
  // member: blue_cones
  {
    if (msg.blue_cones.size() == 0) {
      out << "blue_cones: []";
    } else {
      out << "blue_cones: [";
      size_t pending_items = msg.blue_cones.size();
      for (auto item : msg.blue_cones) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: yellow_cones
  {
    if (msg.yellow_cones.size() == 0) {
      out << "yellow_cones: []";
    } else {
      out << "yellow_cones: [";
      size_t pending_items = msg.yellow_cones.size();
      for (auto item : msg.yellow_cones) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: orange_cones
  {
    if (msg.orange_cones.size() == 0) {
      out << "orange_cones: []";
    } else {
      out << "orange_cones: [";
      size_t pending_items = msg.orange_cones.size();
      for (auto item : msg.orange_cones) {
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
  const ConeList & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: blue_cones
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.blue_cones.size() == 0) {
      out << "blue_cones: []\n";
    } else {
      out << "blue_cones:\n";
      for (auto item : msg.blue_cones) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: yellow_cones
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.yellow_cones.size() == 0) {
      out << "yellow_cones: []\n";
    } else {
      out << "yellow_cones:\n";
      for (auto item : msg.yellow_cones) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: orange_cones
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.orange_cones.size() == 0) {
      out << "orange_cones: []\n";
    } else {
      out << "orange_cones:\n";
      for (auto item : msg.orange_cones) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const ConeList & msg, bool use_flow_style = false)
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
  const interfaces::msg::ConeList & msg,
  std::ostream & out, size_t indentation = 0)
{
  interfaces::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use interfaces::msg::to_yaml() instead")]]
inline std::string to_yaml(const interfaces::msg::ConeList & msg)
{
  return interfaces::msg::to_yaml(msg);
}

template<>
inline const char * data_type<interfaces::msg::ConeList>()
{
  return "interfaces::msg::ConeList";
}

template<>
inline const char * name<interfaces::msg::ConeList>()
{
  return "interfaces/msg/ConeList";
}

template<>
struct has_fixed_size<interfaces::msg::ConeList>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<interfaces::msg::ConeList>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<interfaces::msg::ConeList>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // INTERFACES__MSG__DETAIL__CONE_LIST__TRAITS_HPP_
