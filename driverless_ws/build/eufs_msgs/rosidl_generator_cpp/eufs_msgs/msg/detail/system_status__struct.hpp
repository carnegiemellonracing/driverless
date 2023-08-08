// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/SystemStatus.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__SYSTEM_STATUS__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__SYSTEM_STATUS__STRUCT_HPP_

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
// Member 'topic_statuses'
#include "eufs_msgs/msg/detail/topic_status__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__SystemStatus __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__SystemStatus __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SystemStatus_
{
  using Type = SystemStatus_<ContainerAllocator>;

  explicit SystemStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit SystemStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _topic_statuses_type =
    std::vector<eufs_msgs::msg::TopicStatus_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::TopicStatus_<ContainerAllocator>>::other>;
  _topic_statuses_type topic_statuses;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__topic_statuses(
    const std::vector<eufs_msgs::msg::TopicStatus_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::TopicStatus_<ContainerAllocator>>::other> & _arg)
  {
    this->topic_statuses = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::SystemStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::SystemStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::SystemStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::SystemStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::SystemStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::SystemStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::SystemStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::SystemStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::SystemStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::SystemStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__SystemStatus
    std::shared_ptr<eufs_msgs::msg::SystemStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__SystemStatus
    std::shared_ptr<eufs_msgs::msg::SystemStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SystemStatus_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->topic_statuses != other.topic_statuses) {
      return false;
    }
    return true;
  }
  bool operator!=(const SystemStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SystemStatus_

// alias to use template instance with default allocator
using SystemStatus =
  eufs_msgs::msg::SystemStatus_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__SYSTEM_STATUS__STRUCT_HPP_
