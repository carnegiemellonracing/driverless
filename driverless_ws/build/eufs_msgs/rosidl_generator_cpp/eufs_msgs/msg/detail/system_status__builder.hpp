// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/SystemStatus.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__SYSTEM_STATUS__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__SYSTEM_STATUS__BUILDER_HPP_

#include "eufs_msgs/msg/detail/system_status__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_SystemStatus_topic_statuses
{
public:
  explicit Init_SystemStatus_topic_statuses(::eufs_msgs::msg::SystemStatus & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::SystemStatus topic_statuses(::eufs_msgs::msg::SystemStatus::_topic_statuses_type arg)
  {
    msg_.topic_statuses = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::SystemStatus msg_;
};

class Init_SystemStatus_header
{
public:
  Init_SystemStatus_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SystemStatus_topic_statuses header(::eufs_msgs::msg::SystemStatus::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_SystemStatus_topic_statuses(msg_);
  }

private:
  ::eufs_msgs::msg::SystemStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::SystemStatus>()
{
  return eufs_msgs::msg::builder::Init_SystemStatus_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__SYSTEM_STATUS__BUILDER_HPP_
