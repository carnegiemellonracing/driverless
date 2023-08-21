// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/VehicleCommands.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__BUILDER_HPP_

#include "eufs_msgs/msg/detail/vehicle_commands__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_VehicleCommands_rpm
{
public:
  explicit Init_VehicleCommands_rpm(::eufs_msgs::msg::VehicleCommands & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::VehicleCommands rpm(::eufs_msgs::msg::VehicleCommands::_rpm_type arg)
  {
    msg_.rpm = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::VehicleCommands msg_;
};

class Init_VehicleCommands_steering
{
public:
  explicit Init_VehicleCommands_steering(::eufs_msgs::msg::VehicleCommands & msg)
  : msg_(msg)
  {}
  Init_VehicleCommands_rpm steering(::eufs_msgs::msg::VehicleCommands::_steering_type arg)
  {
    msg_.steering = std::move(arg);
    return Init_VehicleCommands_rpm(msg_);
  }

private:
  ::eufs_msgs::msg::VehicleCommands msg_;
};

class Init_VehicleCommands_torque
{
public:
  explicit Init_VehicleCommands_torque(::eufs_msgs::msg::VehicleCommands & msg)
  : msg_(msg)
  {}
  Init_VehicleCommands_steering torque(::eufs_msgs::msg::VehicleCommands::_torque_type arg)
  {
    msg_.torque = std::move(arg);
    return Init_VehicleCommands_steering(msg_);
  }

private:
  ::eufs_msgs::msg::VehicleCommands msg_;
};

class Init_VehicleCommands_braking
{
public:
  explicit Init_VehicleCommands_braking(::eufs_msgs::msg::VehicleCommands & msg)
  : msg_(msg)
  {}
  Init_VehicleCommands_torque braking(::eufs_msgs::msg::VehicleCommands::_braking_type arg)
  {
    msg_.braking = std::move(arg);
    return Init_VehicleCommands_torque(msg_);
  }

private:
  ::eufs_msgs::msg::VehicleCommands msg_;
};

class Init_VehicleCommands_mission_status
{
public:
  explicit Init_VehicleCommands_mission_status(::eufs_msgs::msg::VehicleCommands & msg)
  : msg_(msg)
  {}
  Init_VehicleCommands_braking mission_status(::eufs_msgs::msg::VehicleCommands::_mission_status_type arg)
  {
    msg_.mission_status = std::move(arg);
    return Init_VehicleCommands_braking(msg_);
  }

private:
  ::eufs_msgs::msg::VehicleCommands msg_;
};

class Init_VehicleCommands_direction
{
public:
  explicit Init_VehicleCommands_direction(::eufs_msgs::msg::VehicleCommands & msg)
  : msg_(msg)
  {}
  Init_VehicleCommands_mission_status direction(::eufs_msgs::msg::VehicleCommands::_direction_type arg)
  {
    msg_.direction = std::move(arg);
    return Init_VehicleCommands_mission_status(msg_);
  }

private:
  ::eufs_msgs::msg::VehicleCommands msg_;
};

class Init_VehicleCommands_ebs
{
public:
  explicit Init_VehicleCommands_ebs(::eufs_msgs::msg::VehicleCommands & msg)
  : msg_(msg)
  {}
  Init_VehicleCommands_direction ebs(::eufs_msgs::msg::VehicleCommands::_ebs_type arg)
  {
    msg_.ebs = std::move(arg);
    return Init_VehicleCommands_direction(msg_);
  }

private:
  ::eufs_msgs::msg::VehicleCommands msg_;
};

class Init_VehicleCommands_handshake
{
public:
  Init_VehicleCommands_handshake()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_VehicleCommands_ebs handshake(::eufs_msgs::msg::VehicleCommands::_handshake_type arg)
  {
    msg_.handshake = std::move(arg);
    return Init_VehicleCommands_ebs(msg_);
  }

private:
  ::eufs_msgs::msg::VehicleCommands msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::VehicleCommands>()
{
  return eufs_msgs::msg::builder::Init_VehicleCommands_handshake();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__BUILDER_HPP_
