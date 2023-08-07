// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/StateMachineState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__STATE_MACHINE_STATE__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__STATE_MACHINE_STATE__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__StateMachineState __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__StateMachineState __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct StateMachineState_
{
  using Type = StateMachineState_<ContainerAllocator>;

  explicit StateMachineState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->state = 0;
      this->state_str = "";
    }
  }

  explicit StateMachineState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : state_str(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->state = 0;
      this->state_str = "";
    }
  }

  // field types and members
  using _state_type =
    uint16_t;
  _state_type state;
  using _state_str_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _state_str_type state_str;

  // setters for named parameter idiom
  Type & set__state(
    const uint16_t & _arg)
  {
    this->state = _arg;
    return *this;
  }
  Type & set__state_str(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->state_str = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::StateMachineState_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::StateMachineState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::StateMachineState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::StateMachineState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::StateMachineState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::StateMachineState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::StateMachineState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::StateMachineState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::StateMachineState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::StateMachineState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__StateMachineState
    std::shared_ptr<eufs_msgs::msg::StateMachineState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__StateMachineState
    std::shared_ptr<eufs_msgs::msg::StateMachineState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const StateMachineState_ & other) const
  {
    if (this->state != other.state) {
      return false;
    }
    if (this->state_str != other.state_str) {
      return false;
    }
    return true;
  }
  bool operator!=(const StateMachineState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct StateMachineState_

// alias to use template instance with default allocator
using StateMachineState =
  eufs_msgs::msg::StateMachineState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__STATE_MACHINE_STATE__STRUCT_HPP_
