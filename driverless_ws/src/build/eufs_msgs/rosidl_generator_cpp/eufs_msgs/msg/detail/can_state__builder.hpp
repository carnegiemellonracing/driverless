// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/CanState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CAN_STATE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__CAN_STATE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/can_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_CanState_ami_state
{
public:
  explicit Init_CanState_ami_state(::eufs_msgs::msg::CanState & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::CanState ami_state(::eufs_msgs::msg::CanState::_ami_state_type arg)
  {
    msg_.ami_state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::CanState msg_;
};

class Init_CanState_as_state
{
public:
  Init_CanState_as_state()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CanState_ami_state as_state(::eufs_msgs::msg::CanState::_as_state_type arg)
  {
    msg_.as_state = std::move(arg);
    return Init_CanState_ami_state(msg_);
  }

private:
  ::eufs_msgs::msg::CanState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::CanState>()
{
  return eufs_msgs::msg::builder::Init_CanState_as_state();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__CAN_STATE__BUILDER_HPP_
