// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/WheelOdometryErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WHEEL_ODOMETRY_ERR__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__WHEEL_ODOMETRY_ERR__BUILDER_HPP_

#include "eufs_msgs/msg/detail/wheel_odometry_err__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_WheelOdometryErr_forward_vel_err
{
public:
  explicit Init_WheelOdometryErr_forward_vel_err(::eufs_msgs::msg::WheelOdometryErr & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::WheelOdometryErr forward_vel_err(::eufs_msgs::msg::WheelOdometryErr::_forward_vel_err_type arg)
  {
    msg_.forward_vel_err = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::WheelOdometryErr msg_;
};

class Init_WheelOdometryErr_angular_vel_err
{
public:
  explicit Init_WheelOdometryErr_angular_vel_err(::eufs_msgs::msg::WheelOdometryErr & msg)
  : msg_(msg)
  {}
  Init_WheelOdometryErr_forward_vel_err angular_vel_err(::eufs_msgs::msg::WheelOdometryErr::_angular_vel_err_type arg)
  {
    msg_.angular_vel_err = std::move(arg);
    return Init_WheelOdometryErr_forward_vel_err(msg_);
  }

private:
  ::eufs_msgs::msg::WheelOdometryErr msg_;
};

class Init_WheelOdometryErr_linear_vel_err
{
public:
  explicit Init_WheelOdometryErr_linear_vel_err(::eufs_msgs::msg::WheelOdometryErr & msg)
  : msg_(msg)
  {}
  Init_WheelOdometryErr_angular_vel_err linear_vel_err(::eufs_msgs::msg::WheelOdometryErr::_linear_vel_err_type arg)
  {
    msg_.linear_vel_err = std::move(arg);
    return Init_WheelOdometryErr_angular_vel_err(msg_);
  }

private:
  ::eufs_msgs::msg::WheelOdometryErr msg_;
};

class Init_WheelOdometryErr_orientation_err
{
public:
  explicit Init_WheelOdometryErr_orientation_err(::eufs_msgs::msg::WheelOdometryErr & msg)
  : msg_(msg)
  {}
  Init_WheelOdometryErr_linear_vel_err orientation_err(::eufs_msgs::msg::WheelOdometryErr::_orientation_err_type arg)
  {
    msg_.orientation_err = std::move(arg);
    return Init_WheelOdometryErr_linear_vel_err(msg_);
  }

private:
  ::eufs_msgs::msg::WheelOdometryErr msg_;
};

class Init_WheelOdometryErr_position_err
{
public:
  explicit Init_WheelOdometryErr_position_err(::eufs_msgs::msg::WheelOdometryErr & msg)
  : msg_(msg)
  {}
  Init_WheelOdometryErr_orientation_err position_err(::eufs_msgs::msg::WheelOdometryErr::_position_err_type arg)
  {
    msg_.position_err = std::move(arg);
    return Init_WheelOdometryErr_orientation_err(msg_);
  }

private:
  ::eufs_msgs::msg::WheelOdometryErr msg_;
};

class Init_WheelOdometryErr_header
{
public:
  Init_WheelOdometryErr_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_WheelOdometryErr_position_err header(::eufs_msgs::msg::WheelOdometryErr::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_WheelOdometryErr_position_err(msg_);
  }

private:
  ::eufs_msgs::msg::WheelOdometryErr msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::WheelOdometryErr>()
{
  return eufs_msgs::msg::builder::Init_WheelOdometryErr_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__WHEEL_ODOMETRY_ERR__BUILDER_HPP_
