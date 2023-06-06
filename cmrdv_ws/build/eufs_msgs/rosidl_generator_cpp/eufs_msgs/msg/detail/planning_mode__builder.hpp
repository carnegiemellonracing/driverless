// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/PlanningMode.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/planning_mode__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_PlanningMode_mode
{
public:
  Init_PlanningMode_mode()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::eufs_msgs::msg::PlanningMode mode(::eufs_msgs::msg::PlanningMode::_mode_type arg)
  {
    msg_.mode = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::PlanningMode msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::PlanningMode>()
{
  return eufs_msgs::msg::builder::Init_PlanningMode_mode();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__BUILDER_HPP_
