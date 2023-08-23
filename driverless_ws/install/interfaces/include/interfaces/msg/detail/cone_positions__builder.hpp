// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from interfaces:msg/ConePositions.idl
// generated code does not contain a copyright notice

#ifndef INTERFACES__MSG__DETAIL__CONE_POSITIONS__BUILDER_HPP_
#define INTERFACES__MSG__DETAIL__CONE_POSITIONS__BUILDER_HPP_

#include "interfaces/msg/detail/cone_positions__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace interfaces
{

namespace msg
{

namespace builder
{

class Init_ConePositions_cone_positions
{
public:
  explicit Init_ConePositions_cone_positions(::interfaces::msg::ConePositions & msg)
  : msg_(msg)
  {}
  ::interfaces::msg::ConePositions cone_positions(::interfaces::msg::ConePositions::_cone_positions_type arg)
  {
    msg_.cone_positions = std::move(arg);
    return std::move(msg_);
  }

private:
  ::interfaces::msg::ConePositions msg_;
};

class Init_ConePositions_header
{
public:
  Init_ConePositions_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ConePositions_cone_positions header(::interfaces::msg::ConePositions::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ConePositions_cone_positions(msg_);
  }

private:
  ::interfaces::msg::ConePositions msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::interfaces::msg::ConePositions>()
{
  return interfaces::msg::builder::Init_ConePositions_header();
}

}  // namespace interfaces

#endif  // INTERFACES__MSG__DETAIL__CONE_POSITIONS__BUILDER_HPP_
