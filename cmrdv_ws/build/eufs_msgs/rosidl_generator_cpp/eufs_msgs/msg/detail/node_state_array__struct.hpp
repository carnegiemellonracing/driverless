// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/NodeStateArray.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__NODE_STATE_ARRAY__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__NODE_STATE_ARRAY__STRUCT_HPP_

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
// Member 'states'
#include "eufs_msgs/msg/detail/node_state__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__NodeStateArray __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__NodeStateArray __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct NodeStateArray_
{
  using Type = NodeStateArray_<ContainerAllocator>;

  explicit NodeStateArray_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit NodeStateArray_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _states_type =
    std::vector<eufs_msgs::msg::NodeState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<eufs_msgs::msg::NodeState_<ContainerAllocator>>>;
  _states_type states;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__states(
    const std::vector<eufs_msgs::msg::NodeState_<ContainerAllocator>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<eufs_msgs::msg::NodeState_<ContainerAllocator>>> & _arg)
  {
    this->states = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::NodeStateArray_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::NodeStateArray_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::NodeStateArray_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::NodeStateArray_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::NodeStateArray_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::NodeStateArray_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::NodeStateArray_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::NodeStateArray_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::NodeStateArray_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::NodeStateArray_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__NodeStateArray
    std::shared_ptr<eufs_msgs::msg::NodeStateArray_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__NodeStateArray
    std::shared_ptr<eufs_msgs::msg::NodeStateArray_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const NodeStateArray_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->states != other.states) {
      return false;
    }
    return true;
  }
  bool operator!=(const NodeStateArray_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct NodeStateArray_

// alias to use template instance with default allocator
using NodeStateArray =
  eufs_msgs::msg::NodeStateArray_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__NODE_STATE_ARRAY__STRUCT_HPP_
