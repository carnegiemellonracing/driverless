// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cmrdv_interfaces:msg/ConeList.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CONE_LIST__STRUCT_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__CONE_LIST__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'blue_cones'
// Member 'yellow_cones'
// Member 'orange_cones'
#include "geometry_msgs/msg/detail/point__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__cmrdv_interfaces__msg__ConeList __attribute__((deprecated))
#else
# define DEPRECATED__cmrdv_interfaces__msg__ConeList __declspec(deprecated)
#endif

namespace cmrdv_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ConeList_
{
  using Type = ConeList_<ContainerAllocator>;

  explicit ConeList_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit ConeList_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _blue_cones_type =
    std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename ContainerAllocator::template rebind<geometry_msgs::msg::Point_<ContainerAllocator>>::other>;
  _blue_cones_type blue_cones;
  using _yellow_cones_type =
    std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename ContainerAllocator::template rebind<geometry_msgs::msg::Point_<ContainerAllocator>>::other>;
  _yellow_cones_type yellow_cones;
  using _orange_cones_type =
    std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename ContainerAllocator::template rebind<geometry_msgs::msg::Point_<ContainerAllocator>>::other>;
  _orange_cones_type orange_cones;

  // setters for named parameter idiom
  Type & set__blue_cones(
    const std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename ContainerAllocator::template rebind<geometry_msgs::msg::Point_<ContainerAllocator>>::other> & _arg)
  {
    this->blue_cones = _arg;
    return *this;
  }
  Type & set__yellow_cones(
    const std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename ContainerAllocator::template rebind<geometry_msgs::msg::Point_<ContainerAllocator>>::other> & _arg)
  {
    this->yellow_cones = _arg;
    return *this;
  }
  Type & set__orange_cones(
    const std::vector<geometry_msgs::msg::Point_<ContainerAllocator>, typename ContainerAllocator::template rebind<geometry_msgs::msg::Point_<ContainerAllocator>>::other> & _arg)
  {
    this->orange_cones = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cmrdv_interfaces::msg::ConeList_<ContainerAllocator> *;
  using ConstRawPtr =
    const cmrdv_interfaces::msg::ConeList_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::ConeList_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::ConeList_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::ConeList_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::ConeList_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::ConeList_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::ConeList_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::ConeList_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::ConeList_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cmrdv_interfaces__msg__ConeList
    std::shared_ptr<cmrdv_interfaces::msg::ConeList_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cmrdv_interfaces__msg__ConeList
    std::shared_ptr<cmrdv_interfaces::msg::ConeList_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ConeList_ & other) const
  {
    if (this->blue_cones != other.blue_cones) {
      return false;
    }
    if (this->yellow_cones != other.yellow_cones) {
      return false;
    }
    if (this->orange_cones != other.orange_cones) {
      return false;
    }
    return true;
  }
  bool operator!=(const ConeList_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ConeList_

// alias to use template instance with default allocator
using ConeList =
  cmrdv_interfaces::msg::ConeList_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CONE_LIST__STRUCT_HPP_