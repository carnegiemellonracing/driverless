// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/ConeArrayWithCovariance.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__TRAITS_HPP_

#include "eufs_msgs/msg/detail/cone_array_with_covariance__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'blue_cones'
// Member 'yellow_cones'
// Member 'orange_cones'
// Member 'big_orange_cones'
// Member 'unknown_color_cones'
#include "eufs_msgs/msg/detail/cone_with_covariance__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::ConeArrayWithCovariance & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_yaml(msg.header, out, indentation + 2);
  }

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
        to_yaml(item, out, indentation + 2);
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
        to_yaml(item, out, indentation + 2);
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
        to_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: big_orange_cones
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.big_orange_cones.size() == 0) {
      out << "big_orange_cones: []\n";
    } else {
      out << "big_orange_cones:\n";
      for (auto item : msg.big_orange_cones) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_yaml(item, out, indentation + 2);
      }
    }
  }

  // member: unknown_color_cones
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.unknown_color_cones.size() == 0) {
      out << "unknown_color_cones: []\n";
    } else {
      out << "unknown_color_cones:\n";
      for (auto item : msg.unknown_color_cones) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::ConeArrayWithCovariance & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

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
