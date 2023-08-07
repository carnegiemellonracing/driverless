// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/WheelSpeedsStamped.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS_STAMPED__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS_STAMPED__BUILDER_HPP_

#include "eufs_msgs/msg/detail/wheel_speeds_stamped__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_WheelSpeedsStamped_speeds
{
public:
  explicit Init_WheelSpeedsStamped_speeds(::eufs_msgs::msg::WheelSpeedsStamped & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::WheelSpeedsStamped speeds(::eufs_msgs::msg::WheelSpeedsStamped::_speeds_type arg)
  {
    msg_.speeds = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::WheelSpeedsStamped msg_;
};

class Init_WheelSpeedsStamped_header
{
public:
  Init_WheelSpeedsStamped_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_WheelSpeedsStamped_speeds header(::eufs_msgs::msg::WheelSpeedsStamped::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_WheelSpeedsStamped_speeds(msg_);
  }

private:
  ::eufs_msgs::msg::WheelSpeedsStamped msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::WheelSpeedsStamped>()
{
  return eufs_msgs::msg::builder::Init_WheelSpeedsStamped_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS_STAMPED__BUILDER_HPP_
