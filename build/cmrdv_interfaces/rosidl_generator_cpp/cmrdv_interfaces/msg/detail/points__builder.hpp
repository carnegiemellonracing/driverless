// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cmrdv_interfaces:msg/Points.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__POINTS__BUILDER_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__POINTS__BUILDER_HPP_

#include "cmrdv_interfaces/msg/detail/points__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace cmrdv_interfaces
{

namespace msg
{

namespace builder
{

class Init_Points_points
{
public:
  explicit Init_Points_points(::cmrdv_interfaces::msg::Points & msg)
  : msg_(msg)
  {}
  ::cmrdv_interfaces::msg::Points points(::cmrdv_interfaces::msg::Points::_points_type arg)
  {
    msg_.points = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cmrdv_interfaces::msg::Points msg_;
};

class Init_Points_header
{
public:
  Init_Points_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Points_points header(::cmrdv_interfaces::msg::Points::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_Points_points(msg_);
  }

private:
  ::cmrdv_interfaces::msg::Points msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cmrdv_interfaces::msg::Points>()
{
  return cmrdv_interfaces::msg::builder::Init_Points_header();
}

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__POINTS__BUILDER_HPP_
