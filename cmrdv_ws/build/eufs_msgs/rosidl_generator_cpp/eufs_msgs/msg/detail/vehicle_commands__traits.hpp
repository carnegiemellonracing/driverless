// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/VehicleCommands.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__TRAITS_HPP_

#include "eufs_msgs/msg/detail/vehicle_commands__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::VehicleCommands & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: handshake
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "handshake: ";
    value_to_yaml(msg.handshake, out);
    out << "\n";
  }

  // member: ebs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ebs: ";
    value_to_yaml(msg.ebs, out);
    out << "\n";
  }

  // member: direction
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "direction: ";
    value_to_yaml(msg.direction, out);
    out << "\n";
  }

  // member: mission_status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "mission_status: ";
    value_to_yaml(msg.mission_status, out);
    out << "\n";
  }

  // member: braking
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "braking: ";
    value_to_yaml(msg.braking, out);
    out << "\n";
  }

  // member: torque
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "torque: ";
    value_to_yaml(msg.torque, out);
    out << "\n";
  }

  // member: steering
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "steering: ";
    value_to_yaml(msg.steering, out);
    out << "\n";
  }

  // member: rpm
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "rpm: ";
    value_to_yaml(msg.rpm, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::VehicleCommands & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::VehicleCommands>()
{
  return "eufs_msgs::msg::VehicleCommands";
}

template<>
inline const char * name<eufs_msgs::msg::VehicleCommands>()
{
  return "eufs_msgs/msg/VehicleCommands";
}

template<>
struct has_fixed_size<eufs_msgs::msg::VehicleCommands>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<eufs_msgs::msg::VehicleCommands>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<eufs_msgs::msg::VehicleCommands>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__TRAITS_HPP_
