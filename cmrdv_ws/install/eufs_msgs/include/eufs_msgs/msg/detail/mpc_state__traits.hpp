// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/MPCState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__MPC_STATE__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__MPC_STATE__TRAITS_HPP_

#include "eufs_msgs/msg/detail/mpc_state__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::MPCState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: exitflag
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "exitflag: ";
    value_to_yaml(msg.exitflag, out);
    out << "\n";
  }

  // member: iterations
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "iterations: ";
    value_to_yaml(msg.iterations, out);
    out << "\n";
  }

  // member: solve_time
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "solve_time: ";
    value_to_yaml(msg.solve_time, out);
    out << "\n";
  }

  // member: cost
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "cost: ";
    value_to_yaml(msg.cost, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::MPCState & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::MPCState>()
{
  return "eufs_msgs::msg::MPCState";
}

template<>
inline const char * name<eufs_msgs::msg::MPCState>()
{
  return "eufs_msgs/msg/MPCState";
}

template<>
struct has_fixed_size<eufs_msgs::msg::MPCState>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<eufs_msgs::msg::MPCState>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<eufs_msgs::msg::MPCState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__MPC_STATE__TRAITS_HPP_
