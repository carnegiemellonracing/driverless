// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/Waypoint.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WAYPOINT__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__WAYPOINT__BUILDER_HPP_

#include "eufs_msgs/msg/detail/waypoint__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_Waypoint_suggested_steering
{
public:
  explicit Init_Waypoint_suggested_steering(::eufs_msgs::msg::Waypoint & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::Waypoint suggested_steering(::eufs_msgs::msg::Waypoint::_suggested_steering_type arg)
  {
    msg_.suggested_steering = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::Waypoint msg_;
};

class Init_Waypoint_speed
{
public:
  explicit Init_Waypoint_speed(::eufs_msgs::msg::Waypoint & msg)
  : msg_(msg)
  {}
  Init_Waypoint_suggested_steering speed(::eufs_msgs::msg::Waypoint::_speed_type arg)
  {
    msg_.speed = std::move(arg);
    return Init_Waypoint_suggested_steering(msg_);
  }

private:
  ::eufs_msgs::msg::Waypoint msg_;
};

class Init_Waypoint_position
{
public:
  Init_Waypoint_position()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Waypoint_speed position(::eufs_msgs::msg::Waypoint::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_Waypoint_speed(msg_);
  }

private:
  ::eufs_msgs::msg::Waypoint msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::Waypoint>()
{
  return eufs_msgs::msg::builder::Init_Waypoint_position();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__WAYPOINT__BUILDER_HPP_
