// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/PathIntegralStats.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__TRAITS_HPP_

#include "eufs_msgs/msg/detail/path_integral_stats__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'params'
#include "eufs_msgs/msg/detail/path_integral_params__traits.hpp"
// Member 'stats'
#include "eufs_msgs/msg/detail/lap_stats__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::PathIntegralStats & msg,
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

  // member: tag
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "tag: ";
    value_to_yaml(msg.tag, out);
    out << "\n";
  }

  // member: params
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "params:\n";
    to_yaml(msg.params, out, indentation + 2);
  }

  // member: stats
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stats:\n";
    to_yaml(msg.stats, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::PathIntegralStats & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::PathIntegralStats>()
{
  return "eufs_msgs::msg::PathIntegralStats";
}

template<>
inline const char * name<eufs_msgs::msg::PathIntegralStats>()
{
  return "eufs_msgs/msg/PathIntegralStats";
}

template<>
struct has_fixed_size<eufs_msgs::msg::PathIntegralStats>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<eufs_msgs::msg::PathIntegralStats>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<eufs_msgs::msg::PathIntegralStats>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__TRAITS_HPP_
