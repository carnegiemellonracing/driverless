// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/EKFState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__EKF_STATE__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__EKF_STATE__TRAITS_HPP_

#include "eufs_msgs/msg/detail/ekf_state__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::EKFState & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: gps_received
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "gps_received: ";
    value_to_yaml(msg.gps_received, out);
    out << "\n";
  }

  // member: imu_received
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "imu_received: ";
    value_to_yaml(msg.imu_received, out);
    out << "\n";
  }

  // member: wheel_odom_received
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "wheel_odom_received: ";
    value_to_yaml(msg.wheel_odom_received, out);
    out << "\n";
  }

  // member: ekf_odom_received
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ekf_odom_received: ";
    value_to_yaml(msg.ekf_odom_received, out);
    out << "\n";
  }

  // member: ekf_accel_received
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ekf_accel_received: ";
    value_to_yaml(msg.ekf_accel_received, out);
    out << "\n";
  }

  // member: currently_over_covariance_limit
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "currently_over_covariance_limit: ";
    value_to_yaml(msg.currently_over_covariance_limit, out);
    out << "\n";
  }

  // member: consecutive_turns_over_covariance_limit
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "consecutive_turns_over_covariance_limit: ";
    value_to_yaml(msg.consecutive_turns_over_covariance_limit, out);
    out << "\n";
  }

  // member: recommends_failure
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "recommends_failure: ";
    value_to_yaml(msg.recommends_failure, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::EKFState & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::EKFState>()
{
  return "eufs_msgs::msg::EKFState";
}

template<>
inline const char * name<eufs_msgs::msg::EKFState>()
{
  return "eufs_msgs/msg/EKFState";
}

template<>
struct has_fixed_size<eufs_msgs::msg::EKFState>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<eufs_msgs::msg::EKFState>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<eufs_msgs::msg::EKFState>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__EKF_STATE__TRAITS_HPP_
