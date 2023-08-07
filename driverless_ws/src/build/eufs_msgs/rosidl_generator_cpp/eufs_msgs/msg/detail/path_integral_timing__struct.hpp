// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/PathIntegralTiming.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_TIMING__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_TIMING__STRUCT_HPP_

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
# define DEPRECATED__eufs_msgs__msg__PathIntegralTiming __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__PathIntegralTiming __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PathIntegralTiming_
{
  using Type = PathIntegralTiming_<ContainerAllocator>;

  explicit PathIntegralTiming_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->average_time_between_poses = 0.0;
      this->average_optimization_cycle_time = 0.0;
      this->average_sleep_time = 0.0;
    }
  }

  explicit PathIntegralTiming_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->average_time_between_poses = 0.0;
      this->average_optimization_cycle_time = 0.0;
      this->average_sleep_time = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _average_time_between_poses_type =
    double;
  _average_time_between_poses_type average_time_between_poses;
  using _average_optimization_cycle_time_type =
    double;
  _average_optimization_cycle_time_type average_optimization_cycle_time;
  using _average_sleep_time_type =
    double;
  _average_sleep_time_type average_sleep_time;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__average_time_between_poses(
    const double & _arg)
  {
    this->average_time_between_poses = _arg;
    return *this;
  }
  Type & set__average_optimization_cycle_time(
    const double & _arg)
  {
    this->average_optimization_cycle_time = _arg;
    return *this;
  }
  Type & set__average_sleep_time(
    const double & _arg)
  {
    this->average_sleep_time = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__PathIntegralTiming
    std::shared_ptr<eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__PathIntegralTiming
    std::shared_ptr<eufs_msgs::msg::PathIntegralTiming_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PathIntegralTiming_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->average_time_between_poses != other.average_time_between_poses) {
      return false;
    }
    if (this->average_optimization_cycle_time != other.average_optimization_cycle_time) {
      return false;
    }
    if (this->average_sleep_time != other.average_sleep_time) {
      return false;
    }
    return true;
  }
  bool operator!=(const PathIntegralTiming_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PathIntegralTiming_

// alias to use template instance with default allocator
using PathIntegralTiming =
  eufs_msgs::msg::PathIntegralTiming_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_TIMING__STRUCT_HPP_
