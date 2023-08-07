// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/WheelSpeeds.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__BUILDER_HPP_

#include "eufs_msgs/msg/detail/wheel_speeds__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_WheelSpeeds_rb_speed
{
public:
  explicit Init_WheelSpeeds_rb_speed(::eufs_msgs::msg::WheelSpeeds & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::WheelSpeeds rb_speed(::eufs_msgs::msg::WheelSpeeds::_rb_speed_type arg)
  {
    msg_.rb_speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::WheelSpeeds msg_;
};

class Init_WheelSpeeds_lb_speed
{
public:
  explicit Init_WheelSpeeds_lb_speed(::eufs_msgs::msg::WheelSpeeds & msg)
  : msg_(msg)
  {}
  Init_WheelSpeeds_rb_speed lb_speed(::eufs_msgs::msg::WheelSpeeds::_lb_speed_type arg)
  {
    msg_.lb_speed = std::move(arg);
    return Init_WheelSpeeds_rb_speed(msg_);
  }

private:
  ::eufs_msgs::msg::WheelSpeeds msg_;
};

class Init_WheelSpeeds_rf_speed
{
public:
  explicit Init_WheelSpeeds_rf_speed(::eufs_msgs::msg::WheelSpeeds & msg)
  : msg_(msg)
  {}
  Init_WheelSpeeds_lb_speed rf_speed(::eufs_msgs::msg::WheelSpeeds::_rf_speed_type arg)
  {
    msg_.rf_speed = std::move(arg);
    return Init_WheelSpeeds_lb_speed(msg_);
  }

private:
  ::eufs_msgs::msg::WheelSpeeds msg_;
};

class Init_WheelSpeeds_lf_speed
{
public:
  explicit Init_WheelSpeeds_lf_speed(::eufs_msgs::msg::WheelSpeeds & msg)
  : msg_(msg)
  {}
  Init_WheelSpeeds_rf_speed lf_speed(::eufs_msgs::msg::WheelSpeeds::_lf_speed_type arg)
  {
    msg_.lf_speed = std::move(arg);
    return Init_WheelSpeeds_rf_speed(msg_);
  }

private:
  ::eufs_msgs::msg::WheelSpeeds msg_;
};

class Init_WheelSpeeds_steering
{
public:
  Init_WheelSpeeds_steering()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_WheelSpeeds_lf_speed steering(::eufs_msgs::msg::WheelSpeeds::_steering_type arg)
  {
    msg_.steering = std::move(arg);
    return Init_WheelSpeeds_lf_speed(msg_);
  }

private:
  ::eufs_msgs::msg::WheelSpeeds msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::WheelSpeeds>()
{
  return eufs_msgs::msg::builder::Init_WheelSpeeds_steering();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__BUILDER_HPP_
