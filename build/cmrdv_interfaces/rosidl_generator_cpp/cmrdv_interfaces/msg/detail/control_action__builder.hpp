// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cmrdv_interfaces:msg/ControlAction.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__BUILDER_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__BUILDER_HPP_

#include "cmrdv_interfaces/msg/detail/control_action__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace cmrdv_interfaces
{

namespace msg
{

namespace builder
{

class Init_ControlAction_swangle
{
public:
  explicit Init_ControlAction_swangle(::cmrdv_interfaces::msg::ControlAction & msg)
  : msg_(msg)
  {}
  ::cmrdv_interfaces::msg::ControlAction swangle(::cmrdv_interfaces::msg::ControlAction::_swangle_type arg)
  {
    msg_.swangle = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cmrdv_interfaces::msg::ControlAction msg_;
};

class Init_ControlAction_wheel_speed
{
public:
  Init_ControlAction_wheel_speed()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ControlAction_swangle wheel_speed(::cmrdv_interfaces::msg::ControlAction::_wheel_speed_type arg)
  {
    msg_.wheel_speed = std::move(arg);
    return Init_ControlAction_swangle(msg_);
  }

private:
  ::cmrdv_interfaces::msg::ControlAction msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cmrdv_interfaces::msg::ControlAction>()
{
  return cmrdv_interfaces::msg::builder::Init_ControlAction_wheel_speed();
}

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__BUILDER_HPP_
