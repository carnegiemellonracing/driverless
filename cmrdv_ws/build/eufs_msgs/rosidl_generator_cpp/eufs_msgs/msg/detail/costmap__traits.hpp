// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/Costmap.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__COSTMAP__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__COSTMAP__TRAITS_HPP_

#include "eufs_msgs/msg/detail/costmap__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::Costmap & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: x_bounds_min
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x_bounds_min: ";
    value_to_yaml(msg.x_bounds_min, out);
    out << "\n";
  }

  // member: x_bounds_max
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "x_bounds_max: ";
    value_to_yaml(msg.x_bounds_max, out);
    out << "\n";
  }

  // member: y_bounds_min
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y_bounds_min: ";
    value_to_yaml(msg.y_bounds_min, out);
    out << "\n";
  }

  // member: y_bounds_max
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "y_bounds_max: ";
    value_to_yaml(msg.y_bounds_max, out);
    out << "\n";
  }

  // member: pixels_per_meter
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "pixels_per_meter: ";
    value_to_yaml(msg.pixels_per_meter, out);
    out << "\n";
  }

  // member: channel0
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.channel0.size() == 0) {
      out << "channel0: []\n";
    } else {
      out << "channel0:\n";
      for (auto item : msg.channel0) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: channel1
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.channel1.size() == 0) {
      out << "channel1: []\n";
    } else {
      out << "channel1:\n";
      for (auto item : msg.channel1) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: channel2
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.channel2.size() == 0) {
      out << "channel2: []\n";
    } else {
      out << "channel2:\n";
      for (auto item : msg.channel2) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: channel3
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.channel3.size() == 0) {
      out << "channel3: []\n";
    } else {
      out << "channel3:\n";
      for (auto item : msg.channel3) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::Costmap & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::Costmap>()
{
  return "eufs_msgs::msg::Costmap";
}

template<>
inline const char * name<eufs_msgs::msg::Costmap>()
{
  return "eufs_msgs/msg/Costmap";
}

template<>
struct has_fixed_size<eufs_msgs::msg::Costmap>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<eufs_msgs::msg::Costmap>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<eufs_msgs::msg::Costmap>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__COSTMAP__TRAITS_HPP_
