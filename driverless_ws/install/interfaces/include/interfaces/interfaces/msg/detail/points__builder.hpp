// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from interfaces:msg/Points.idl
// generated code does not contain a copyright notice

#ifndef INTERFACES__MSG__DETAIL__POINTS__BUILDER_HPP_
#define INTERFACES__MSG__DETAIL__POINTS__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "interfaces/msg/detail/points__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace interfaces
{

namespace msg
{

namespace builder
{

class Init_Points_points
{
public:
  explicit Init_Points_points(::interfaces::msg::Points & msg)
  : msg_(msg)
  {}
  ::interfaces::msg::Points points(::interfaces::msg::Points::_points_type arg)
  {
    msg_.points = std::move(arg);
    return std::move(msg_);
  }

private:
  ::interfaces::msg::Points msg_;
};

class Init_Points_header
{
public:
  Init_Points_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Points_points header(::interfaces::msg::Points::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_Points_points(msg_);
  }

private:
  ::interfaces::msg::Points msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::interfaces::msg::Points>()
{
  return interfaces::msg::builder::Init_Points_header();
}

}  // namespace interfaces

#endif  // INTERFACES__MSG__DETAIL__POINTS__BUILDER_HPP_
