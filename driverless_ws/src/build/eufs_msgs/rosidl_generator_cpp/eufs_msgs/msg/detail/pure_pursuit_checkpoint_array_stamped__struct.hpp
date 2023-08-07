// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/PurePursuitCheckpointArrayStamped.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT_ARRAY_STAMPED__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT_ARRAY_STAMPED__STRUCT_HPP_

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
// Member 'checkpoints'
#include "eufs_msgs/msg/detail/pure_pursuit_checkpoint__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__PurePursuitCheckpointArrayStamped __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__PurePursuitCheckpointArrayStamped __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PurePursuitCheckpointArrayStamped_
{
  using Type = PurePursuitCheckpointArrayStamped_<ContainerAllocator>;

  explicit PurePursuitCheckpointArrayStamped_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit PurePursuitCheckpointArrayStamped_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _checkpoints_type =
    std::vector<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator>>::other>;
  _checkpoints_type checkpoints;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__checkpoints(
    const std::vector<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator>>::other> & _arg)
  {
    this->checkpoints = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__PurePursuitCheckpointArrayStamped
    std::shared_ptr<eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__PurePursuitCheckpointArrayStamped
    std::shared_ptr<eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PurePursuitCheckpointArrayStamped_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->checkpoints != other.checkpoints) {
      return false;
    }
    return true;
  }
  bool operator!=(const PurePursuitCheckpointArrayStamped_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PurePursuitCheckpointArrayStamped_

// alias to use template instance with default allocator
using PurePursuitCheckpointArrayStamped =
  eufs_msgs::msg::PurePursuitCheckpointArrayStamped_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT_ARRAY_STAMPED__STRUCT_HPP_
