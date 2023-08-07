// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/NodeStateArray.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__NODE_STATE_ARRAY__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__NODE_STATE_ARRAY__BUILDER_HPP_

#include "eufs_msgs/msg/detail/node_state_array__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_NodeStateArray_states
{
public:
  explicit Init_NodeStateArray_states(::eufs_msgs::msg::NodeStateArray & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::NodeStateArray states(::eufs_msgs::msg::NodeStateArray::_states_type arg)
  {
    msg_.states = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::NodeStateArray msg_;
};

class Init_NodeStateArray_header
{
public:
  Init_NodeStateArray_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_NodeStateArray_states header(::eufs_msgs::msg::NodeStateArray::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_NodeStateArray_states(msg_);
  }

private:
  ::eufs_msgs::msg::NodeStateArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::NodeStateArray>()
{
  return eufs_msgs::msg::builder::Init_NodeStateArray_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__NODE_STATE_ARRAY__BUILDER_HPP_
