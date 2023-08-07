// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/FullState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__FULL_STATE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__FULL_STATE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/full_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_FullState_steering
{
public:
  explicit Init_FullState_steering(::eufs_msgs::msg::FullState & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::FullState steering(::eufs_msgs::msg::FullState::_steering_type arg)
  {
    msg_.steering = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::FullState msg_;
};

class Init_FullState_rear_throttle
{
public:
  explicit Init_FullState_rear_throttle(::eufs_msgs::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_steering rear_throttle(::eufs_msgs::msg::FullState::_rear_throttle_type arg)
  {
    msg_.rear_throttle = std::move(arg);
    return Init_FullState_steering(msg_);
  }

private:
  ::eufs_msgs::msg::FullState msg_;
};

class Init_FullState_front_throttle
{
public:
  explicit Init_FullState_front_throttle(::eufs_msgs::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_rear_throttle front_throttle(::eufs_msgs::msg::FullState::_front_throttle_type arg)
  {
    msg_.front_throttle = std::move(arg);
    return Init_FullState_rear_throttle(msg_);
  }

private:
  ::eufs_msgs::msg::FullState msg_;
};

class Init_FullState_yaw_mder
{
public:
  explicit Init_FullState_yaw_mder(::eufs_msgs::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_front_throttle yaw_mder(::eufs_msgs::msg::FullState::_yaw_mder_type arg)
  {
    msg_.yaw_mder = std::move(arg);
    return Init_FullState_front_throttle(msg_);
  }

private:
  ::eufs_msgs::msg::FullState msg_;
};

class Init_FullState_u_y
{
public:
  explicit Init_FullState_u_y(::eufs_msgs::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_yaw_mder u_y(::eufs_msgs::msg::FullState::_u_y_type arg)
  {
    msg_.u_y = std::move(arg);
    return Init_FullState_yaw_mder(msg_);
  }

private:
  ::eufs_msgs::msg::FullState msg_;
};

class Init_FullState_u_x
{
public:
  explicit Init_FullState_u_x(::eufs_msgs::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_u_y u_x(::eufs_msgs::msg::FullState::_u_x_type arg)
  {
    msg_.u_x = std::move(arg);
    return Init_FullState_u_y(msg_);
  }

private:
  ::eufs_msgs::msg::FullState msg_;
};

class Init_FullState_roll
{
public:
  explicit Init_FullState_roll(::eufs_msgs::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_u_x roll(::eufs_msgs::msg::FullState::_roll_type arg)
  {
    msg_.roll = std::move(arg);
    return Init_FullState_u_x(msg_);
  }

private:
  ::eufs_msgs::msg::FullState msg_;
};

class Init_FullState_yaw
{
public:
  explicit Init_FullState_yaw(::eufs_msgs::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_roll yaw(::eufs_msgs::msg::FullState::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_FullState_roll(msg_);
  }

private:
  ::eufs_msgs::msg::FullState msg_;
};

class Init_FullState_y_pos
{
public:
  explicit Init_FullState_y_pos(::eufs_msgs::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_yaw y_pos(::eufs_msgs::msg::FullState::_y_pos_type arg)
  {
    msg_.y_pos = std::move(arg);
    return Init_FullState_yaw(msg_);
  }

private:
  ::eufs_msgs::msg::FullState msg_;
};

class Init_FullState_x_pos
{
public:
  explicit Init_FullState_x_pos(::eufs_msgs::msg::FullState & msg)
  : msg_(msg)
  {}
  Init_FullState_y_pos x_pos(::eufs_msgs::msg::FullState::_x_pos_type arg)
  {
    msg_.x_pos = std::move(arg);
    return Init_FullState_y_pos(msg_);
  }

private:
  ::eufs_msgs::msg::FullState msg_;
};

class Init_FullState_header
{
public:
  Init_FullState_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_FullState_x_pos header(::eufs_msgs::msg::FullState::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_FullState_x_pos(msg_);
  }

private:
  ::eufs_msgs::msg::FullState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::FullState>()
{
  return eufs_msgs::msg::builder::Init_FullState_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__FULL_STATE__BUILDER_HPP_
