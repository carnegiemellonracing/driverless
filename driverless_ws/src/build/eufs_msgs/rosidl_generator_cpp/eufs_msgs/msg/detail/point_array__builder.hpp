// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/PointArray.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__POINT_ARRAY__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__POINT_ARRAY__BUILDER_HPP_

#include "eufs_msgs/msg/detail/point_array__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_PointArray_points
{
public:
  Init_PointArray_points()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::eufs_msgs::msg::PointArray points(::eufs_msgs::msg::PointArray::_points_type arg)
  {
    msg_.points = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::PointArray msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::PointArray>()
{
  return eufs_msgs::msg::builder::Init_PointArray_points();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__POINT_ARRAY__BUILDER_HPP_
