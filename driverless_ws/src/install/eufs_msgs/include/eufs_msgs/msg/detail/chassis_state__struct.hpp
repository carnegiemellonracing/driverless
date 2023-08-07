// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/ChassisState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CHASSIS_STATE__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__CHASSIS_STATE__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__ChassisState __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__ChassisState __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ChassisState_
{
  using Type = ChassisState_<ContainerAllocator>;

  explicit ChassisState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->throttle_relay_enabled = false;
      this->autonomous_enabled = false;
      this->runstop_motion_enabled = false;
      this->steering_commander = "";
      this->steering = 0.0;
      this->throttle_commander = "";
      this->throttle = 0.0;
      this->front_brake_commander = "";
      this->front_brake = 0.0;
    }
  }

  explicit ChassisState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    steering_commander(_alloc),
    throttle_commander(_alloc),
    front_brake_commander(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->throttle_relay_enabled = false;
      this->autonomous_enabled = false;
      this->runstop_motion_enabled = false;
      this->steering_commander = "";
      this->steering = 0.0;
      this->throttle_commander = "";
      this->throttle = 0.0;
      this->front_brake_commander = "";
      this->front_brake = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _throttle_relay_enabled_type =
    bool;
  _throttle_relay_enabled_type throttle_relay_enabled;
  using _autonomous_enabled_type =
    bool;
  _autonomous_enabled_type autonomous_enabled;
  using _runstop_motion_enabled_type =
    bool;
  _runstop_motion_enabled_type runstop_motion_enabled;
  using _steering_commander_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _steering_commander_type steering_commander;
  using _steering_type =
    double;
  _steering_type steering;
  using _throttle_commander_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _throttle_commander_type throttle_commander;
  using _throttle_type =
    double;
  _throttle_type throttle;
  using _front_brake_commander_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _front_brake_commander_type front_brake_commander;
  using _front_brake_type =
    double;
  _front_brake_type front_brake;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__throttle_relay_enabled(
    const bool & _arg)
  {
    this->throttle_relay_enabled = _arg;
    return *this;
  }
  Type & set__autonomous_enabled(
    const bool & _arg)
  {
    this->autonomous_enabled = _arg;
    return *this;
  }
  Type & set__runstop_motion_enabled(
    const bool & _arg)
  {
    this->runstop_motion_enabled = _arg;
    return *this;
  }
  Type & set__steering_commander(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->steering_commander = _arg;
    return *this;
  }
  Type & set__steering(
    const double & _arg)
  {
    this->steering = _arg;
    return *this;
  }
  Type & set__throttle_commander(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->throttle_commander = _arg;
    return *this;
  }
  Type & set__throttle(
    const double & _arg)
  {
    this->throttle = _arg;
    return *this;
  }
  Type & set__front_brake_commander(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->front_brake_commander = _arg;
    return *this;
  }
  Type & set__front_brake(
    const double & _arg)
  {
    this->front_brake = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::ChassisState_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::ChassisState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::ChassisState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::ChassisState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::ChassisState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::ChassisState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::ChassisState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::ChassisState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::ChassisState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::ChassisState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__ChassisState
    std::shared_ptr<eufs_msgs::msg::ChassisState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__ChassisState
    std::shared_ptr<eufs_msgs::msg::ChassisState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ChassisState_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->throttle_relay_enabled != other.throttle_relay_enabled) {
      return false;
    }
    if (this->autonomous_enabled != other.autonomous_enabled) {
      return false;
    }
    if (this->runstop_motion_enabled != other.runstop_motion_enabled) {
      return false;
    }
    if (this->steering_commander != other.steering_commander) {
      return false;
    }
    if (this->steering != other.steering) {
      return false;
    }
    if (this->throttle_commander != other.throttle_commander) {
      return false;
    }
    if (this->throttle != other.throttle) {
      return false;
    }
    if (this->front_brake_commander != other.front_brake_commander) {
      return false;
    }
    if (this->front_brake != other.front_brake) {
      return false;
    }
    return true;
  }
  bool operator!=(const ChassisState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ChassisState_

// alias to use template instance with default allocator
using ChassisState =
  eufs_msgs::msg::ChassisState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__CHASSIS_STATE__STRUCT_HPP_
