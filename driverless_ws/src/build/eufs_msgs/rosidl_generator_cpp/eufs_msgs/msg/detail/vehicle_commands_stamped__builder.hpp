// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/VehicleCommandsStamped.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS_STAMPED__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS_STAMPED__BUILDER_HPP_

#include "eufs_msgs/msg/detail/vehicle_commands_stamped__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_VehicleCommandsStamped_commands
{
public:
  explicit Init_VehicleCommandsStamped_commands(::eufs_msgs::msg::VehicleCommandsStamped & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::VehicleCommandsStamped commands(::eufs_msgs::msg::VehicleCommandsStamped::_commands_type arg)
  {
    msg_.commands = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::VehicleCommandsStamped msg_;
};

class Init_VehicleCommandsStamped_header
{
public:
  Init_VehicleCommandsStamped_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_VehicleCommandsStamped_commands header(::eufs_msgs::msg::VehicleCommandsStamped::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_VehicleCommandsStamped_commands(msg_);
  }

private:
  ::eufs_msgs::msg::VehicleCommandsStamped msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::VehicleCommandsStamped>()
{
  return eufs_msgs::msg::builder::Init_VehicleCommandsStamped_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS_STAMPED__BUILDER_HPP_
