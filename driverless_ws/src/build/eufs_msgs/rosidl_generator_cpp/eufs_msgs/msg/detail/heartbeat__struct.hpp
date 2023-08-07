// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/Heartbeat.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__HEARTBEAT__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__HEARTBEAT__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__Heartbeat __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__Heartbeat __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Heartbeat_
{
  using Type = Heartbeat_<ContainerAllocator>;

  explicit Heartbeat_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
      this->data = 0;
    }
  }

  explicit Heartbeat_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
      this->data = 0;
    }
  }

  // field types and members
  using _id_type =
    uint8_t;
  _id_type id;
  using _data_type =
    uint8_t;
  _data_type data;

  // setters for named parameter idiom
  Type & set__id(
    const uint8_t & _arg)
  {
    this->id = _arg;
    return *this;
  }
  Type & set__data(
    const uint8_t & _arg)
  {
    this->data = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::Heartbeat_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::Heartbeat_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::Heartbeat_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::Heartbeat_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::Heartbeat_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::Heartbeat_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::Heartbeat_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::Heartbeat_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::Heartbeat_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::Heartbeat_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__Heartbeat
    std::shared_ptr<eufs_msgs::msg::Heartbeat_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__Heartbeat
    std::shared_ptr<eufs_msgs::msg::Heartbeat_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Heartbeat_ & other) const
  {
    if (this->id != other.id) {
      return false;
    }
    if (this->data != other.data) {
      return false;
    }
    return true;
  }
  bool operator!=(const Heartbeat_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Heartbeat_

// alias to use template instance with default allocator
using Heartbeat =
  eufs_msgs::msg::Heartbeat_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__HEARTBEAT__STRUCT_HPP_
