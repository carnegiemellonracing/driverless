// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/NodeState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__NODE_STATE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__NODE_STATE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/node_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_NodeState_online
{
public:
  explicit Init_NodeState_online(::eufs_msgs::msg::NodeState & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::NodeState online(::eufs_msgs::msg::NodeState::_online_type arg)
  {
    msg_.online = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::NodeState msg_;
};

class Init_NodeState_severity
{
public:
  explicit Init_NodeState_severity(::eufs_msgs::msg::NodeState & msg)
  : msg_(msg)
  {}
  Init_NodeState_online severity(::eufs_msgs::msg::NodeState::_severity_type arg)
  {
    msg_.severity = std::move(arg);
    return Init_NodeState_online(msg_);
  }

private:
  ::eufs_msgs::msg::NodeState msg_;
};

class Init_NodeState_last_heartbeat
{
public:
  explicit Init_NodeState_last_heartbeat(::eufs_msgs::msg::NodeState & msg)
  : msg_(msg)
  {}
  Init_NodeState_severity last_heartbeat(::eufs_msgs::msg::NodeState::_last_heartbeat_type arg)
  {
    msg_.last_heartbeat = std::move(arg);
    return Init_NodeState_severity(msg_);
  }

private:
  ::eufs_msgs::msg::NodeState msg_;
};

class Init_NodeState_exp_heartbeat
{
public:
  explicit Init_NodeState_exp_heartbeat(::eufs_msgs::msg::NodeState & msg)
  : msg_(msg)
  {}
  Init_NodeState_last_heartbeat exp_heartbeat(::eufs_msgs::msg::NodeState::_exp_heartbeat_type arg)
  {
    msg_.exp_heartbeat = std::move(arg);
    return Init_NodeState_last_heartbeat(msg_);
  }

private:
  ::eufs_msgs::msg::NodeState msg_;
};

class Init_NodeState_name
{
public:
  explicit Init_NodeState_name(::eufs_msgs::msg::NodeState & msg)
  : msg_(msg)
  {}
  Init_NodeState_exp_heartbeat name(::eufs_msgs::msg::NodeState::_name_type arg)
  {
    msg_.name = std::move(arg);
    return Init_NodeState_exp_heartbeat(msg_);
  }

private:
  ::eufs_msgs::msg::NodeState msg_;
};

class Init_NodeState_id
{
public:
  Init_NodeState_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NodeState_name id(::eufs_msgs::msg::NodeState::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_NodeState_name(msg_);
  }

private:
  ::eufs_msgs::msg::NodeState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::NodeState>()
{
  return eufs_msgs::msg::builder::Init_NodeState_id();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__NODE_STATE__BUILDER_HPP_
