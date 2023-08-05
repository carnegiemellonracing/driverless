// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from interfaces:msg/PairROT.idl
// generated code does not contain a copyright notice

#ifndef INTERFACES__MSG__DETAIL__PAIR_ROT__BUILDER_HPP_
#define INTERFACES__MSG__DETAIL__PAIR_ROT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "interfaces/msg/detail/pair_rot__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace interfaces
{

namespace msg
{

namespace builder
{

class Init_PairROT_far
{
public:
  explicit Init_PairROT_far(::interfaces::msg::PairROT & msg)
  : msg_(msg)
  {}
  ::interfaces::msg::PairROT far(::interfaces::msg::PairROT::_far_type arg)
  {
    msg_.far = std::move(arg);
    return std::move(msg_);
  }

private:
  ::interfaces::msg::PairROT msg_;
};

class Init_PairROT_near
{
public:
  explicit Init_PairROT_near(::interfaces::msg::PairROT & msg)
  : msg_(msg)
  {}
  Init_PairROT_far near(::interfaces::msg::PairROT::_near_type arg)
  {
    msg_.near = std::move(arg);
    return Init_PairROT_far(msg_);
  }

private:
  ::interfaces::msg::PairROT msg_;
};

class Init_PairROT_header
{
public:
  Init_PairROT_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PairROT_near header(::interfaces::msg::PairROT::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_PairROT_near(msg_);
  }

private:
  ::interfaces::msg::PairROT msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::interfaces::msg::PairROT>()
{
  return interfaces::msg::builder::Init_PairROT_header();
}

}  // namespace interfaces

#endif  // INTERFACES__MSG__DETAIL__PAIR_ROT__BUILDER_HPP_
