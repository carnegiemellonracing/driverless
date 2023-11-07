// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/PurePursuitCheckpointArrayStamped.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT_ARRAY_STAMPED__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT_ARRAY_STAMPED__BUILDER_HPP_

#include "eufs_msgs/msg/detail/pure_pursuit_checkpoint_array_stamped__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_PurePursuitCheckpointArrayStamped_checkpoints
{
public:
  explicit Init_PurePursuitCheckpointArrayStamped_checkpoints(::eufs_msgs::msg::PurePursuitCheckpointArrayStamped & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::PurePursuitCheckpointArrayStamped checkpoints(::eufs_msgs::msg::PurePursuitCheckpointArrayStamped::_checkpoints_type arg)
  {
    msg_.checkpoints = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::PurePursuitCheckpointArrayStamped msg_;
};

class Init_PurePursuitCheckpointArrayStamped_header
{
public:
  Init_PurePursuitCheckpointArrayStamped_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PurePursuitCheckpointArrayStamped_checkpoints header(::eufs_msgs::msg::PurePursuitCheckpointArrayStamped::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_PurePursuitCheckpointArrayStamped_checkpoints(msg_);
  }

private:
  ::eufs_msgs::msg::PurePursuitCheckpointArrayStamped msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::PurePursuitCheckpointArrayStamped>()
{
  return eufs_msgs::msg::builder::Init_PurePursuitCheckpointArrayStamped_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT_ARRAY_STAMPED__BUILDER_HPP_
