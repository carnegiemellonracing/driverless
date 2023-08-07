// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/ConeArrayWithCovariance.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__STRUCT_HPP_

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
// Member 'blue_cones'
// Member 'yellow_cones'
// Member 'orange_cones'
// Member 'big_orange_cones'
// Member 'unknown_color_cones'
#include "eufs_msgs/msg/detail/cone_with_covariance__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__ConeArrayWithCovariance __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__ConeArrayWithCovariance __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ConeArrayWithCovariance_
{
  using Type = ConeArrayWithCovariance_<ContainerAllocator>;

  explicit ConeArrayWithCovariance_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit ConeArrayWithCovariance_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _blue_cones_type =
    std::vector<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>>::other>;
  _blue_cones_type blue_cones;
  using _yellow_cones_type =
    std::vector<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>>::other>;
  _yellow_cones_type yellow_cones;
  using _orange_cones_type =
    std::vector<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>>::other>;
  _orange_cones_type orange_cones;
  using _big_orange_cones_type =
    std::vector<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>>::other>;
  _big_orange_cones_type big_orange_cones;
  using _unknown_color_cones_type =
    std::vector<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>>::other>;
  _unknown_color_cones_type unknown_color_cones;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__blue_cones(
    const std::vector<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>>::other> & _arg)
  {
    this->blue_cones = _arg;
    return *this;
  }
  Type & set__yellow_cones(
    const std::vector<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>>::other> & _arg)
  {
    this->yellow_cones = _arg;
    return *this;
  }
  Type & set__orange_cones(
    const std::vector<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>>::other> & _arg)
  {
    this->orange_cones = _arg;
    return *this;
  }
  Type & set__big_orange_cones(
    const std::vector<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>>::other> & _arg)
  {
    this->big_orange_cones = _arg;
    return *this;
  }
  Type & set__unknown_color_cones(
    const std::vector<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::ConeWithCovariance_<ContainerAllocator>>::other> & _arg)
  {
    this->unknown_color_cones = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__ConeArrayWithCovariance
    std::shared_ptr<eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__ConeArrayWithCovariance
    std::shared_ptr<eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ConeArrayWithCovariance_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->blue_cones != other.blue_cones) {
      return false;
    }
    if (this->yellow_cones != other.yellow_cones) {
      return false;
    }
    if (this->orange_cones != other.orange_cones) {
      return false;
    }
    if (this->big_orange_cones != other.big_orange_cones) {
      return false;
    }
    if (this->unknown_color_cones != other.unknown_color_cones) {
      return false;
    }
    return true;
  }
  bool operator!=(const ConeArrayWithCovariance_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ConeArrayWithCovariance_

// alias to use template instance with default allocator
using ConeArrayWithCovariance =
  eufs_msgs::msg::ConeArrayWithCovariance_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__STRUCT_HPP_
