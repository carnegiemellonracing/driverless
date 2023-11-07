// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/PurePursuitCheckpoint.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__BUILDER_HPP_

#include "eufs_msgs/msg/detail/pure_pursuit_checkpoint__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_PurePursuitCheckpoint_max_lateral_acceleration
{
public:
  explicit Init_PurePursuitCheckpoint_max_lateral_acceleration(::eufs_msgs::msg::PurePursuitCheckpoint & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::PurePursuitCheckpoint max_lateral_acceleration(::eufs_msgs::msg::PurePursuitCheckpoint::_max_lateral_acceleration_type arg)
  {
    msg_.max_lateral_acceleration = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::PurePursuitCheckpoint msg_;
};

class Init_PurePursuitCheckpoint_max_speed
{
public:
  explicit Init_PurePursuitCheckpoint_max_speed(::eufs_msgs::msg::PurePursuitCheckpoint & msg)
  : msg_(msg)
  {}
  Init_PurePursuitCheckpoint_max_lateral_acceleration max_speed(::eufs_msgs::msg::PurePursuitCheckpoint::_max_speed_type arg)
  {
    msg_.max_speed = std::move(arg);
    return Init_PurePursuitCheckpoint_max_lateral_acceleration(msg_);
  }

private:
  ::eufs_msgs::msg::PurePursuitCheckpoint msg_;
};

class Init_PurePursuitCheckpoint_position
{
public:
  Init_PurePursuitCheckpoint_position()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PurePursuitCheckpoint_max_speed position(::eufs_msgs::msg::PurePursuitCheckpoint::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_PurePursuitCheckpoint_max_speed(msg_);
  }

private:
  ::eufs_msgs::msg::PurePursuitCheckpoint msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::PurePursuitCheckpoint>()
{
  return eufs_msgs::msg::builder::Init_PurePursuitCheckpoint_position();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__BUILDER_HPP_
