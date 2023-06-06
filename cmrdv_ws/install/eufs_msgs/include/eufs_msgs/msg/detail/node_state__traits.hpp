// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/NodeState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__NODE_STATE__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__NODE_STATE__TRAITS_HPP_

#include "eufs_msgs/msg/detail/node_state__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

// Include directives for member types
// Member 'last_heartbeat'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::NodeState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    value_to_yaml(msg.id, out);
    out << "\n";
  }

  // member: name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "name: ";
    value_to_yaml(msg.name, out);
    out << "\n";
  }

  // member: exp_heartbeat
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "exp_heartbeat: ";
    value_to_yaml(msg.exp_heartbeat, out);
    out << "\n";
  }

  // member: last_heartbeat
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "last_heartbeat:\n";
    to_yaml(msg.last_heartbeat, out, indentation + 2);
  }

  // member: severity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "severity: ";
    value_to_yaml(msg.severity, out);
    out << "\n";
  }

  // member: online
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "online: ";
    value_to_yaml(msg.online, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::NodeState & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::NodeState>()
{
  return "eufs_msgs::msg::NodeState";
}

template<>
inline const char * name<eufs_msgs::msg::NodeState>()
{
  return "eufs_msgs/msg/NodeState";
}

template<>
struct has_fixed_size<eufs_msgs::msg::NodeState>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<eufs_msgs::msg::NodeState>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<eufs_msgs::msg::NodeState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__NODE_STATE__TRAITS_HPP_
