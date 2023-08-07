// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/ChassisState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CHASSIS_STATE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__CHASSIS_STATE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/chassis_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_ChassisState_front_brake
{
public:
  explicit Init_ChassisState_front_brake(::eufs_msgs::msg::ChassisState & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::ChassisState front_brake(::eufs_msgs::msg::ChassisState::_front_brake_type arg)
  {
    msg_.front_brake = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisState msg_;
};

class Init_ChassisState_front_brake_commander
{
public:
  explicit Init_ChassisState_front_brake_commander(::eufs_msgs::msg::ChassisState & msg)
  : msg_(msg)
  {}
  Init_ChassisState_front_brake front_brake_commander(::eufs_msgs::msg::ChassisState::_front_brake_commander_type arg)
  {
    msg_.front_brake_commander = std::move(arg);
    return Init_ChassisState_front_brake(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisState msg_;
};

class Init_ChassisState_throttle
{
public:
  explicit Init_ChassisState_throttle(::eufs_msgs::msg::ChassisState & msg)
  : msg_(msg)
  {}
  Init_ChassisState_front_brake_commander throttle(::eufs_msgs::msg::ChassisState::_throttle_type arg)
  {
    msg_.throttle = std::move(arg);
    return Init_ChassisState_front_brake_commander(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisState msg_;
};

class Init_ChassisState_throttle_commander
{
public:
  explicit Init_ChassisState_throttle_commander(::eufs_msgs::msg::ChassisState & msg)
  : msg_(msg)
  {}
  Init_ChassisState_throttle throttle_commander(::eufs_msgs::msg::ChassisState::_throttle_commander_type arg)
  {
    msg_.throttle_commander = std::move(arg);
    return Init_ChassisState_throttle(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisState msg_;
};

class Init_ChassisState_steering
{
public:
  explicit Init_ChassisState_steering(::eufs_msgs::msg::ChassisState & msg)
  : msg_(msg)
  {}
  Init_ChassisState_throttle_commander steering(::eufs_msgs::msg::ChassisState::_steering_type arg)
  {
    msg_.steering = std::move(arg);
    return Init_ChassisState_throttle_commander(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisState msg_;
};

class Init_ChassisState_steering_commander
{
public:
  explicit Init_ChassisState_steering_commander(::eufs_msgs::msg::ChassisState & msg)
  : msg_(msg)
  {}
  Init_ChassisState_steering steering_commander(::eufs_msgs::msg::ChassisState::_steering_commander_type arg)
  {
    msg_.steering_commander = std::move(arg);
    return Init_ChassisState_steering(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisState msg_;
};

class Init_ChassisState_runstop_motion_enabled
{
public:
  explicit Init_ChassisState_runstop_motion_enabled(::eufs_msgs::msg::ChassisState & msg)
  : msg_(msg)
  {}
  Init_ChassisState_steering_commander runstop_motion_enabled(::eufs_msgs::msg::ChassisState::_runstop_motion_enabled_type arg)
  {
    msg_.runstop_motion_enabled = std::move(arg);
    return Init_ChassisState_steering_commander(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisState msg_;
};

class Init_ChassisState_autonomous_enabled
{
public:
  explicit Init_ChassisState_autonomous_enabled(::eufs_msgs::msg::ChassisState & msg)
  : msg_(msg)
  {}
  Init_ChassisState_runstop_motion_enabled autonomous_enabled(::eufs_msgs::msg::ChassisState::_autonomous_enabled_type arg)
  {
    msg_.autonomous_enabled = std::move(arg);
    return Init_ChassisState_runstop_motion_enabled(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisState msg_;
};

class Init_ChassisState_throttle_relay_enabled
{
public:
  explicit Init_ChassisState_throttle_relay_enabled(::eufs_msgs::msg::ChassisState & msg)
  : msg_(msg)
  {}
  Init_ChassisState_autonomous_enabled throttle_relay_enabled(::eufs_msgs::msg::ChassisState::_throttle_relay_enabled_type arg)
  {
    msg_.throttle_relay_enabled = std::move(arg);
    return Init_ChassisState_autonomous_enabled(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisState msg_;
};

class Init_ChassisState_header
{
public:
  Init_ChassisState_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ChassisState_throttle_relay_enabled header(::eufs_msgs::msg::ChassisState::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ChassisState_throttle_relay_enabled(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::ChassisState>()
{
  return eufs_msgs::msg::builder::Init_ChassisState_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__CHASSIS_STATE__BUILDER_HPP_
