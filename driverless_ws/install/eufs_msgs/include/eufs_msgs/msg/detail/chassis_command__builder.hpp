// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/ChassisCommand.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CHASSIS_COMMAND__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__CHASSIS_COMMAND__BUILDER_HPP_

#include "eufs_msgs/msg/detail/chassis_command__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_ChassisCommand_front_brake
{
public:
  explicit Init_ChassisCommand_front_brake(::eufs_msgs::msg::ChassisCommand & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::ChassisCommand front_brake(::eufs_msgs::msg::ChassisCommand::_front_brake_type arg)
  {
    msg_.front_brake = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisCommand msg_;
};

class Init_ChassisCommand_steering
{
public:
  explicit Init_ChassisCommand_steering(::eufs_msgs::msg::ChassisCommand & msg)
  : msg_(msg)
  {}
  Init_ChassisCommand_front_brake steering(::eufs_msgs::msg::ChassisCommand::_steering_type arg)
  {
    msg_.steering = std::move(arg);
    return Init_ChassisCommand_front_brake(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisCommand msg_;
};

class Init_ChassisCommand_throttle
{
public:
  explicit Init_ChassisCommand_throttle(::eufs_msgs::msg::ChassisCommand & msg)
  : msg_(msg)
  {}
  Init_ChassisCommand_steering throttle(::eufs_msgs::msg::ChassisCommand::_throttle_type arg)
  {
    msg_.throttle = std::move(arg);
    return Init_ChassisCommand_steering(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisCommand msg_;
};

class Init_ChassisCommand_sender
{
public:
  explicit Init_ChassisCommand_sender(::eufs_msgs::msg::ChassisCommand & msg)
  : msg_(msg)
  {}
  Init_ChassisCommand_throttle sender(::eufs_msgs::msg::ChassisCommand::_sender_type arg)
  {
    msg_.sender = std::move(arg);
    return Init_ChassisCommand_throttle(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisCommand msg_;
};

class Init_ChassisCommand_header
{
public:
  Init_ChassisCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ChassisCommand_sender header(::eufs_msgs::msg::ChassisCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ChassisCommand_sender(msg_);
  }

private:
  ::eufs_msgs::msg::ChassisCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::ChassisCommand>()
{
  return eufs_msgs::msg::builder::Init_ChassisCommand_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__CHASSIS_COMMAND__BUILDER_HPP_
