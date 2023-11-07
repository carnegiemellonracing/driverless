// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cmrdv_interfaces:msg/Heartbeat.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__HEARTBEAT__BUILDER_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__HEARTBEAT__BUILDER_HPP_

#include "cmrdv_interfaces/msg/detail/heartbeat__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace cmrdv_interfaces
{

namespace msg
{

namespace builder
{

class Init_Heartbeat_status
{
public:
  explicit Init_Heartbeat_status(::cmrdv_interfaces::msg::Heartbeat & msg)
  : msg_(msg)
  {}
  ::cmrdv_interfaces::msg::Heartbeat status(::cmrdv_interfaces::msg::Heartbeat::_status_type arg)
  {
    msg_.status = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cmrdv_interfaces::msg::Heartbeat msg_;
};

class Init_Heartbeat_header
{
public:
  Init_Heartbeat_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Heartbeat_status header(::cmrdv_interfaces::msg::Heartbeat::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_Heartbeat_status(msg_);
  }

private:
  ::cmrdv_interfaces::msg::Heartbeat msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cmrdv_interfaces::msg::Heartbeat>()
{
  return cmrdv_interfaces::msg::builder::Init_Heartbeat_header();
}

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__HEARTBEAT__BUILDER_HPP_
