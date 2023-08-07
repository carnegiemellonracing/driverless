// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/PurePursuitCheckpoint.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__PurePursuitCheckpoint __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__PurePursuitCheckpoint __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PurePursuitCheckpoint_
{
  using Type = PurePursuitCheckpoint_<ContainerAllocator>;

  explicit PurePursuitCheckpoint_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->max_speed = 0.0;
      this->max_lateral_acceleration = 0.0;
    }
  }

  explicit PurePursuitCheckpoint_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->max_speed = 0.0;
      this->max_lateral_acceleration = 0.0;
    }
  }

  // field types and members
  using _position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _position_type position;
  using _max_speed_type =
    double;
  _max_speed_type max_speed;
  using _max_lateral_acceleration_type =
    double;
  _max_lateral_acceleration_type max_lateral_acceleration;

  // setters for named parameter idiom
  Type & set__position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__max_speed(
    const double & _arg)
  {
    this->max_speed = _arg;
    return *this;
  }
  Type & set__max_lateral_acceleration(
    const double & _arg)
  {
    this->max_lateral_acceleration = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__PurePursuitCheckpoint
    std::shared_ptr<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__PurePursuitCheckpoint
    std::shared_ptr<eufs_msgs::msg::PurePursuitCheckpoint_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PurePursuitCheckpoint_ & other) const
  {
    if (this->position != other.position) {
      return false;
    }
    if (this->max_speed != other.max_speed) {
      return false;
    }
    if (this->max_lateral_acceleration != other.max_lateral_acceleration) {
      return false;
    }
    return true;
  }
  bool operator!=(const PurePursuitCheckpoint_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PurePursuitCheckpoint_

// alias to use template instance with default allocator
using PurePursuitCheckpoint =
  eufs_msgs::msg::PurePursuitCheckpoint_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__STRUCT_HPP_
