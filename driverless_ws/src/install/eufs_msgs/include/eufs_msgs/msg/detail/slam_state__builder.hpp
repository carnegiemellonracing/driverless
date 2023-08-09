// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/SLAMState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__SLAM_STATE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__SLAM_STATE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/slam_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_SLAMState_state
{
public:
  explicit Init_SLAMState_state(::eufs_msgs::msg::SLAMState & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::SLAMState state(::eufs_msgs::msg::SLAMState::_state_type arg)
  {
    msg_.state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMState msg_;
};

class Init_SLAMState_status
{
public:
  explicit Init_SLAMState_status(::eufs_msgs::msg::SLAMState & msg)
  : msg_(msg)
  {}
  Init_SLAMState_state status(::eufs_msgs::msg::SLAMState::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_SLAMState_state(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMState msg_;
};

class Init_SLAMState_laps
{
public:
  explicit Init_SLAMState_laps(::eufs_msgs::msg::SLAMState & msg)
  : msg_(msg)
  {}
  Init_SLAMState_status laps(::eufs_msgs::msg::SLAMState::_laps_type arg)
  {
    msg_.laps = std::move(arg);
    return Init_SLAMState_status(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMState msg_;
};

class Init_SLAMState_loop_closed
{
public:
  Init_SLAMState_loop_closed()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SLAMState_laps loop_closed(::eufs_msgs::msg::SLAMState::_loop_closed_type arg)
  {
    msg_.loop_closed = std::move(arg);
    return Init_SLAMState_laps(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::SLAMState>()
{
  return eufs_msgs::msg::builder::Init_SLAMState_loop_closed();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__SLAM_STATE__BUILDER_HPP_
