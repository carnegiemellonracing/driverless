// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/Runstop.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__RUNSTOP__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__RUNSTOP__BUILDER_HPP_

#include "eufs_msgs/msg/detail/runstop__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_Runstop_motion_enabled
{
public:
  explicit Init_Runstop_motion_enabled(::eufs_msgs::msg::Runstop & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::Runstop motion_enabled(::eufs_msgs::msg::Runstop::_motion_enabled_type arg)
  {
    msg_.motion_enabled = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::Runstop msg_;
};

class Init_Runstop_sender
{
public:
  explicit Init_Runstop_sender(::eufs_msgs::msg::Runstop & msg)
  : msg_(msg)
  {}
  Init_Runstop_motion_enabled sender(::eufs_msgs::msg::Runstop::_sender_type arg)
  {
    msg_.sender = std::move(arg);
    return Init_Runstop_motion_enabled(msg_);
  }

private:
  ::eufs_msgs::msg::Runstop msg_;
};

class Init_Runstop_header
{
public:
  Init_Runstop_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Runstop_sender header(::eufs_msgs::msg::Runstop::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_Runstop_sender(msg_);
  }

private:
  ::eufs_msgs::msg::Runstop msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::Runstop>()
{
  return eufs_msgs::msg::builder::Init_Runstop_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__RUNSTOP__BUILDER_HPP_
