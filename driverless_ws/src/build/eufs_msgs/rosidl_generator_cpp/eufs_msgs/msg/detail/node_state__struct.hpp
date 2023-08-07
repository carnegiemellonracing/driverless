// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/NodeState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__NODE_STATE__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__NODE_STATE__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'last_heartbeat'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__NodeState __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__NodeState __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct NodeState_
{
  using Type = NodeState_<ContainerAllocator>;

  explicit NodeState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : last_heartbeat(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
      this->name = "";
      this->exp_heartbeat = 0;
      this->severity = 0;
      this->online = false;
    }
  }

  explicit NodeState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : name(_alloc),
    last_heartbeat(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
      this->name = "";
      this->exp_heartbeat = 0;
      this->severity = 0;
      this->online = false;
    }
  }

  // field types and members
  using _id_type =
    uint16_t;
  _id_type id;
  using _name_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _name_type name;
  using _exp_heartbeat_type =
    uint8_t;
  _exp_heartbeat_type exp_heartbeat;
  using _last_heartbeat_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _last_heartbeat_type last_heartbeat;
  using _severity_type =
    uint8_t;
  _severity_type severity;
  using _online_type =
    bool;
  _online_type online;

  // setters for named parameter idiom
  Type & set__id(
    const uint16_t & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__name(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->name = _arg;
    return *this;
  }
  Type & set__exp_heartbeat(
    const uint8_t & _arg)
  {
    this->exp_heartbeat = _arg;
    return *this;
  }
  Type & set__last_heartbeat(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->last_heartbeat = _arg;
    return *this;
  }
  Type & set__severity(
    const uint8_t & _arg)
  {
    this->severity = _arg;
    return *this;
  }
  Type & set__online(
    const bool & _arg)
  {
    this->online = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::NodeState_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::NodeState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::NodeState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::NodeState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::NodeState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::NodeState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::NodeState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::NodeState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::NodeState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::NodeState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__NodeState
    std::shared_ptr<eufs_msgs::msg::NodeState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__NodeState
    std::shared_ptr<eufs_msgs::msg::NodeState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NodeState_ & other) const
  {
    if (this->id != other.id) {
      return false;
    }
    if (this->name != other.name) {
      return false;
    }
    if (this->exp_heartbeat != other.exp_heartbeat) {
      return false;
    }
    if (this->last_heartbeat != other.last_heartbeat) {
      return false;
    }
    if (this->severity != other.severity) {
      return false;
    }
    if (this->online != other.online) {
      return false;
    }
    return true;
  }
  bool operator!=(const NodeState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NodeState_

// alias to use template instance with default allocator
using NodeState =
  eufs_msgs::msg::NodeState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__NODE_STATE__STRUCT_HPP_
