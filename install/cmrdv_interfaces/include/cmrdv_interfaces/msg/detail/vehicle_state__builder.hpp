// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cmrdv_interfaces:msg/VehicleState.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__VEHICLE_STATE__BUILDER_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__VEHICLE_STATE__BUILDER_HPP_

#include "cmrdv_interfaces/msg/detail/vehicle_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace cmrdv_interfaces
{

namespace msg
{

namespace builder
{

class Init_VehicleState_acceleration
{
public:
  explicit Init_VehicleState_acceleration(::cmrdv_interfaces::msg::VehicleState & msg)
  : msg_(msg)
  {}
  ::cmrdv_interfaces::msg::VehicleState acceleration(::cmrdv_interfaces::msg::VehicleState::_acceleration_type arg)
  {
    msg_.acceleration = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cmrdv_interfaces::msg::VehicleState msg_;
};

class Init_VehicleState_velocity
{
public:
  explicit Init_VehicleState_velocity(::cmrdv_interfaces::msg::VehicleState & msg)
  : msg_(msg)
  {}
  Init_VehicleState_acceleration velocity(::cmrdv_interfaces::msg::VehicleState::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_VehicleState_acceleration(msg_);
  }

private:
  ::cmrdv_interfaces::msg::VehicleState msg_;
};

class Init_VehicleState_position
{
public:
  explicit Init_VehicleState_position(::cmrdv_interfaces::msg::VehicleState & msg)
  : msg_(msg)
  {}
  Init_VehicleState_velocity position(::cmrdv_interfaces::msg::VehicleState::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_VehicleState_velocity(msg_);
  }

private:
  ::cmrdv_interfaces::msg::VehicleState msg_;
};

class Init_VehicleState_header
{
public:
  Init_VehicleState_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_VehicleState_position header(::cmrdv_interfaces::msg::VehicleState::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_VehicleState_position(msg_);
  }

private:
  ::cmrdv_interfaces::msg::VehicleState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cmrdv_interfaces::msg::VehicleState>()
{
  return cmrdv_interfaces::msg::builder::Init_VehicleState_header();
}

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__VEHICLE_STATE__BUILDER_HPP_
