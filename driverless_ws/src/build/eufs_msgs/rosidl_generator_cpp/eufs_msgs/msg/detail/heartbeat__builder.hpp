// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/Heartbeat.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__HEARTBEAT__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__HEARTBEAT__BUILDER_HPP_

#include "eufs_msgs/msg/detail/heartbeat__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_Heartbeat_data
{
public:
  explicit Init_Heartbeat_data(::eufs_msgs::msg::Heartbeat & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::Heartbeat data(::eufs_msgs::msg::Heartbeat::_data_type arg)
  {
    msg_.data = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::Heartbeat msg_;
};

class Init_Heartbeat_id
{
public:
  Init_Heartbeat_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Heartbeat_data id(::eufs_msgs::msg::Heartbeat::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_Heartbeat_data(msg_);
  }

private:
  ::eufs_msgs::msg::Heartbeat msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::Heartbeat>()
{
  return eufs_msgs::msg::builder::Init_Heartbeat_id();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__HEARTBEAT__BUILDER_HPP_
