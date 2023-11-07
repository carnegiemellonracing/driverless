// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cmrdv_interfaces:msg/PairROT.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__PAIR_ROT__BUILDER_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__PAIR_ROT__BUILDER_HPP_

#include "cmrdv_interfaces/msg/detail/pair_rot__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace cmrdv_interfaces
{

namespace msg
{

namespace builder
{

class Init_PairROT_far
{
public:
  explicit Init_PairROT_far(::cmrdv_interfaces::msg::PairROT & msg)
  : msg_(msg)
  {}
  ::cmrdv_interfaces::msg::PairROT far(::cmrdv_interfaces::msg::PairROT::_far_type arg)
  {
    msg_.far = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cmrdv_interfaces::msg::PairROT msg_;
};

class Init_PairROT_near
{
public:
  explicit Init_PairROT_near(::cmrdv_interfaces::msg::PairROT & msg)
  : msg_(msg)
  {}
  Init_PairROT_far near(::cmrdv_interfaces::msg::PairROT::_near_type arg)
  {
    msg_.near = std::move(arg);
    return Init_PairROT_far(msg_);
  }

private:
  ::cmrdv_interfaces::msg::PairROT msg_;
};

class Init_PairROT_header
{
public:
  Init_PairROT_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PairROT_near header(::cmrdv_interfaces::msg::PairROT::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_PairROT_near(msg_);
  }

private:
  ::cmrdv_interfaces::msg::PairROT msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cmrdv_interfaces::msg::PairROT>()
{
  return cmrdv_interfaces::msg::builder::Init_PairROT_header();
}

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__PAIR_ROT__BUILDER_HPP_
