// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cmrdv_interfaces:msg/Brakes.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__BRAKES__BUILDER_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__BRAKES__BUILDER_HPP_

#include "cmrdv_interfaces/msg/detail/brakes__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace cmrdv_interfaces
{

namespace msg
{

namespace builder
{

class Init_Brakes_last_fired
{
public:
  explicit Init_Brakes_last_fired(::cmrdv_interfaces::msg::Brakes & msg)
  : msg_(msg)
  {}
  ::cmrdv_interfaces::msg::Brakes last_fired(::cmrdv_interfaces::msg::Brakes::_last_fired_type arg)
  {
    msg_.last_fired = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cmrdv_interfaces::msg::Brakes msg_;
};

class Init_Brakes_braking
{
public:
  Init_Brakes_braking()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Brakes_last_fired braking(::cmrdv_interfaces::msg::Brakes::_braking_type arg)
  {
    msg_.braking = std::move(arg);
    return Init_Brakes_last_fired(msg_);
  }

private:
  ::cmrdv_interfaces::msg::Brakes msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cmrdv_interfaces::msg::Brakes>()
{
  return cmrdv_interfaces::msg::builder::Init_Brakes_braking();
}

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__BRAKES__BUILDER_HPP_
