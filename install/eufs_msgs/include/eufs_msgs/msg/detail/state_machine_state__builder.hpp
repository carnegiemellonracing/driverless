// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/StateMachineState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__STATE_MACHINE_STATE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__STATE_MACHINE_STATE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/state_machine_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_StateMachineState_state_str
{
public:
  explicit Init_StateMachineState_state_str(::eufs_msgs::msg::StateMachineState & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::StateMachineState state_str(::eufs_msgs::msg::StateMachineState::_state_str_type arg)
  {
    msg_.state_str = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::StateMachineState msg_;
};

class Init_StateMachineState_state
{
public:
  Init_StateMachineState_state()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_StateMachineState_state_str state(::eufs_msgs::msg::StateMachineState::_state_type arg)
  {
    msg_.state = std::move(arg);
    return Init_StateMachineState_state_str(msg_);
  }

private:
  ::eufs_msgs::msg::StateMachineState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::StateMachineState>()
{
  return eufs_msgs::msg::builder::Init_StateMachineState_state();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__STATE_MACHINE_STATE__BUILDER_HPP_
