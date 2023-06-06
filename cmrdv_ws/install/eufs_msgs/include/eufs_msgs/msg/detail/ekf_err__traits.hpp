// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/EKFErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__EKF_ERR__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__EKF_ERR__TRAITS_HPP_

#include "eufs_msgs/msg/detail/ekf_err__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::EKFErr & msg,
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

  // member: gps_x_vel_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "gps_x_vel_err: ";
    value_to_yaml(msg.gps_x_vel_err, out);
    out << "\n";
  }

  // member: gps_y_vel_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "gps_y_vel_err: ";
    value_to_yaml(msg.gps_y_vel_err, out);
    out << "\n";
  }

  // member: imu_x_acc_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "imu_x_acc_err: ";
    value_to_yaml(msg.imu_x_acc_err, out);
    out << "\n";
  }

  // member: imu_y_acc_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "imu_y_acc_err: ";
    value_to_yaml(msg.imu_y_acc_err, out);
    out << "\n";
  }

  // member: imu_yaw_err
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "imu_yaw_err: ";
    value_to_yaml(msg.imu_yaw_err, out);
    out << "\n";
  }

  // member: ekf_x_vel_var
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ekf_x_vel_var: ";
    value_to_yaml(msg.ekf_x_vel_var, out);
    out << "\n";
  }

  // member: ekf_y_vel_var
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ekf_y_vel_var: ";
    value_to_yaml(msg.ekf_y_vel_var, out);
    out << "\n";
  }

  // member: ekf_x_acc_var
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ekf_x_acc_var: ";
    value_to_yaml(msg.ekf_x_acc_var, out);
    out << "\n";
  }

  // member: ekf_y_acc_var
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ekf_y_acc_var: ";
    value_to_yaml(msg.ekf_y_acc_var, out);
    out << "\n";
  }

  // member: ekf_yaw_var
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ekf_yaw_var: ";
    value_to_yaml(msg.ekf_yaw_var, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::EKFErr & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::EKFErr>()
{
  return "eufs_msgs::msg::EKFErr";
}

template<>
inline const char * name<eufs_msgs::msg::EKFErr>()
{
  return "eufs_msgs/msg/EKFErr";
}

template<>
struct has_fixed_size<eufs_msgs::msg::EKFErr>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<eufs_msgs::msg::EKFErr>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<eufs_msgs::msg::EKFErr>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__EKF_ERR__TRAITS_HPP_
