// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/ChassisCommand.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CHASSIS_COMMAND__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__CHASSIS_COMMAND__STRUCT_HPP_

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
# define DEPRECATED__eufs_msgs__msg__ChassisCommand __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__ChassisCommand __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ChassisCommand_
{
  using Type = ChassisCommand_<ContainerAllocator>;

  explicit ChassisCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->sender = "";
      this->throttle = 0.0;
      this->steering = 0.0;
      this->front_brake = 0.0;
    }
  }

  explicit ChassisCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    sender(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->sender = "";
      this->throttle = 0.0;
      this->steering = 0.0;
      this->front_brake = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _sender_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _sender_type sender;
  using _throttle_type =
    double;
  _throttle_type throttle;
  using _steering_type =
    double;
  _steering_type steering;
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
  Type & set__sender(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->sender = _arg;
    return *this;
  }
  Type & set__throttle(
    const double & _arg)
  {
    this->throttle = _arg;
    return *this;
  }
  Type & set__steering(
    const double & _arg)
  {
    this->steering = _arg;
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
    eufs_msgs::msg::ChassisCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::ChassisCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::ChassisCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::ChassisCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::ChassisCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::ChassisCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::ChassisCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::ChassisCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::ChassisCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::ChassisCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__ChassisCommand
    std::shared_ptr<eufs_msgs::msg::ChassisCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__ChassisCommand
    std::shared_ptr<eufs_msgs::msg::ChassisCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ChassisCommand_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->sender != other.sender) {
      return false;
    }
    if (this->throttle != other.throttle) {
      return false;
    }
    if (this->steering != other.steering) {
      return false;
    }
    if (this->front_brake != other.front_brake) {
      return false;
    }
    return true;
  }
  bool operator!=(const ChassisCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ChassisCommand_

// alias to use template instance with default allocator
using ChassisCommand =
  eufs_msgs::msg::ChassisCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__CHASSIS_COMMAND__STRUCT_HPP_
