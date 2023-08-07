// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/PointArrayStamped.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__POINT_ARRAY_STAMPED__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__POINT_ARRAY_STAMPED__BUILDER_HPP_

#include "eufs_msgs/msg/detail/point_array_stamped__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_PointArrayStamped_points
{
public:
  explicit Init_PointArrayStamped_points(::eufs_msgs::msg::PointArrayStamped & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::PointArrayStamped points(::eufs_msgs::msg::PointArrayStamped::_points_type arg)
  {
    msg_.points = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::PointArrayStamped msg_;
};

class Init_PointArrayStamped_header
{
public:
  Init_PointArrayStamped_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PointArrayStamped_points header(::eufs_msgs::msg::PointArrayStamped::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_PointArrayStamped_points(msg_);
  }

private:
  ::eufs_msgs::msg::PointArrayStamped msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::PointArrayStamped>()
{
  return eufs_msgs::msg::builder::Init_PointArrayStamped_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__POINT_ARRAY_STAMPED__BUILDER_HPP_
