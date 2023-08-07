// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/WaypointArrayStamped.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WAYPOINT_ARRAY_STAMPED__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__WAYPOINT_ARRAY_STAMPED__STRUCT_HPP_

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
// Member 'waypoints'
#include "eufs_msgs/msg/detail/waypoint__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__WaypointArrayStamped __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__WaypointArrayStamped __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct WaypointArrayStamped_
{
  using Type = WaypointArrayStamped_<ContainerAllocator>;

  explicit WaypointArrayStamped_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit WaypointArrayStamped_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _waypoints_type =
    std::vector<eufs_msgs::msg::Waypoint_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::Waypoint_<ContainerAllocator>>::other>;
  _waypoints_type waypoints;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__waypoints(
    const std::vector<eufs_msgs::msg::Waypoint_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::Waypoint_<ContainerAllocator>>::other> & _arg)
  {
    this->waypoints = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__WaypointArrayStamped
    std::shared_ptr<eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__WaypointArrayStamped
    std::shared_ptr<eufs_msgs::msg::WaypointArrayStamped_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const WaypointArrayStamped_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->waypoints != other.waypoints) {
      return false;
    }
    return true;
  }
  bool operator!=(const WaypointArrayStamped_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct WaypointArrayStamped_

// alias to use template instance with default allocator
using WaypointArrayStamped =
  eufs_msgs::msg::WaypointArrayStamped_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__WAYPOINT_ARRAY_STAMPED__STRUCT_HPP_
