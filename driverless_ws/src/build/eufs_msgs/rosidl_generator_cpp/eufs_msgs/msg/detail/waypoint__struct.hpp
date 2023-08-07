// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/Waypoint.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WAYPOINT__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__WAYPOINT__STRUCT_HPP_

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
# define DEPRECATED__eufs_msgs__msg__Waypoint __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__Waypoint __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Waypoint_
{
  using Type = Waypoint_<ContainerAllocator>;

  explicit Waypoint_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->speed = 0.0;
      this->suggested_steering = 0.0;
    }
  }

  explicit Waypoint_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : position(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->speed = 0.0;
      this->suggested_steering = 0.0;
    }
  }

  // field types and members
  using _position_type =
    geometry_msgs::msg::Point_<ContainerAllocator>;
  _position_type position;
  using _speed_type =
    double;
  _speed_type speed;
  using _suggested_steering_type =
    double;
  _suggested_steering_type suggested_steering;

  // setters for named parameter idiom
  Type & set__position(
    const geometry_msgs::msg::Point_<ContainerAllocator> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__speed(
    const double & _arg)
  {
    this->speed = _arg;
    return *this;
  }
  Type & set__suggested_steering(
    const double & _arg)
  {
    this->suggested_steering = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::Waypoint_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::Waypoint_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::Waypoint_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::Waypoint_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::Waypoint_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::Waypoint_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::Waypoint_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::Waypoint_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::Waypoint_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::Waypoint_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__Waypoint
    std::shared_ptr<eufs_msgs::msg::Waypoint_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__Waypoint
    std::shared_ptr<eufs_msgs::msg::Waypoint_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Waypoint_ & other) const
  {
    if (this->position != other.position) {
      return false;
    }
    if (this->speed != other.speed) {
      return false;
    }
    if (this->suggested_steering != other.suggested_steering) {
      return false;
    }
    return true;
  }
  bool operator!=(const Waypoint_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Waypoint_

// alias to use template instance with default allocator
using Waypoint =
  eufs_msgs::msg::Waypoint_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__WAYPOINT__STRUCT_HPP_
