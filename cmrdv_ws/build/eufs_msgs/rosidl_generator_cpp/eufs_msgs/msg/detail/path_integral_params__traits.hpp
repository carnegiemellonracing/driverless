// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/PathIntegralParams.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__TRAITS_HPP_

#include "eufs_msgs/msg/detail/path_integral_params__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::PathIntegralParams & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: hz
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "hz: ";
    value_to_yaml(msg.hz, out);
    out << "\n";
  }

  // member: num_timesteps
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_timesteps: ";
    value_to_yaml(msg.num_timesteps, out);
    out << "\n";
  }

  // member: num_iters
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "num_iters: ";
    value_to_yaml(msg.num_iters, out);
    out << "\n";
  }

  // member: gamma
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "gamma: ";
    value_to_yaml(msg.gamma, out);
    out << "\n";
  }

  // member: init_steering
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "init_steering: ";
    value_to_yaml(msg.init_steering, out);
    out << "\n";
  }

  // member: init_throttle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "init_throttle: ";
    value_to_yaml(msg.init_throttle, out);
    out << "\n";
  }

  // member: steering_var
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "steering_var: ";
    value_to_yaml(msg.steering_var, out);
    out << "\n";
  }

  // member: throttle_var
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "throttle_var: ";
    value_to_yaml(msg.throttle_var, out);
    out << "\n";
  }

  // member: max_throttle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "max_throttle: ";
    value_to_yaml(msg.max_throttle, out);
    out << "\n";
  }

  // member: speed_coefficient
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "speed_coefficient: ";
    value_to_yaml(msg.speed_coefficient, out);
    out << "\n";
  }

  // member: track_coefficient
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "track_coefficient: ";
    value_to_yaml(msg.track_coefficient, out);
    out << "\n";
  }

  // member: max_slip_angle
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "max_slip_angle: ";
    value_to_yaml(msg.max_slip_angle, out);
    out << "\n";
  }

  // member: track_slop
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "track_slop: ";
    value_to_yaml(msg.track_slop, out);
    out << "\n";
  }

  // member: crash_coeff
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "crash_coeff: ";
    value_to_yaml(msg.crash_coeff, out);
    out << "\n";
  }

  // member: map_path
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "map_path: ";
    value_to_yaml(msg.map_path, out);
    out << "\n";
  }

  // member: desired_speed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "desired_speed: ";
    value_to_yaml(msg.desired_speed, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::PathIntegralParams & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::PathIntegralParams>()
{
  return "eufs_msgs::msg::PathIntegralParams";
}

template<>
inline const char * name<eufs_msgs::msg::PathIntegralParams>()
{
  return "eufs_msgs/msg/PathIntegralParams";
}

template<>
struct has_fixed_size<eufs_msgs::msg::PathIntegralParams>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<eufs_msgs::msg::PathIntegralParams>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<eufs_msgs::msg::PathIntegralParams>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__TRAITS_HPP_
