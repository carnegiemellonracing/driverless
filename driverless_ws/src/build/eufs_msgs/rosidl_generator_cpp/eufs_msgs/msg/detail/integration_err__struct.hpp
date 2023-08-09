// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/IntegrationErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__INTEGRATION_ERR__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__INTEGRATION_ERR__STRUCT_HPP_

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
# define DEPRECATED__eufs_msgs__msg__IntegrationErr __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__IntegrationErr __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct IntegrationErr_
{
  using Type = IntegrationErr_<ContainerAllocator>;

  explicit IntegrationErr_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->position_err = 0.0;
      this->orientation_err = 0.0;
      this->linear_vel_err = 0.0;
      this->angular_vel_err = 0.0;
    }
  }

  explicit IntegrationErr_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->position_err = 0.0;
      this->orientation_err = 0.0;
      this->linear_vel_err = 0.0;
      this->angular_vel_err = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _position_err_type =
    double;
  _position_err_type position_err;
  using _orientation_err_type =
    double;
  _orientation_err_type orientation_err;
  using _linear_vel_err_type =
    double;
  _linear_vel_err_type linear_vel_err;
  using _angular_vel_err_type =
    double;
  _angular_vel_err_type angular_vel_err;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__position_err(
    const double & _arg)
  {
    this->position_err = _arg;
    return *this;
  }
  Type & set__orientation_err(
    const double & _arg)
  {
    this->orientation_err = _arg;
    return *this;
  }
  Type & set__linear_vel_err(
    const double & _arg)
  {
    this->linear_vel_err = _arg;
    return *this;
  }
  Type & set__angular_vel_err(
    const double & _arg)
  {
    this->angular_vel_err = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::IntegrationErr_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::IntegrationErr_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::IntegrationErr_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::IntegrationErr_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::IntegrationErr_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::IntegrationErr_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::IntegrationErr_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::IntegrationErr_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::IntegrationErr_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::IntegrationErr_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__IntegrationErr
    std::shared_ptr<eufs_msgs::msg::IntegrationErr_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__IntegrationErr
    std::shared_ptr<eufs_msgs::msg::IntegrationErr_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const IntegrationErr_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->position_err != other.position_err) {
      return false;
    }
    if (this->orientation_err != other.orientation_err) {
      return false;
    }
    if (this->linear_vel_err != other.linear_vel_err) {
      return false;
    }
    if (this->angular_vel_err != other.angular_vel_err) {
      return false;
    }
    return true;
  }
  bool operator!=(const IntegrationErr_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct IntegrationErr_

// alias to use template instance with default allocator
using IntegrationErr =
  eufs_msgs::msg::IntegrationErr_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__INTEGRATION_ERR__STRUCT_HPP_
