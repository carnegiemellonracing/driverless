// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/SLAMErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__SLAM_ERR__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__SLAM_ERR__STRUCT_HPP_

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
# define DEPRECATED__eufs_msgs__msg__SLAMErr __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__SLAMErr __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SLAMErr_
{
  using Type = SLAMErr_<ContainerAllocator>;

  explicit SLAMErr_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x_err = 0.0;
      this->y_err = 0.0;
      this->z_err = 0.0;
      this->x_orient_err = 0.0;
      this->y_orient_err = 0.0;
      this->z_orient_err = 0.0;
      this->w_orient_err = 0.0;
      this->map_similarity = 0.0;
    }
  }

  explicit SLAMErr_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x_err = 0.0;
      this->y_err = 0.0;
      this->z_err = 0.0;
      this->x_orient_err = 0.0;
      this->y_orient_err = 0.0;
      this->z_orient_err = 0.0;
      this->w_orient_err = 0.0;
      this->map_similarity = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _x_err_type =
    double;
  _x_err_type x_err;
  using _y_err_type =
    double;
  _y_err_type y_err;
  using _z_err_type =
    double;
  _z_err_type z_err;
  using _x_orient_err_type =
    double;
  _x_orient_err_type x_orient_err;
  using _y_orient_err_type =
    double;
  _y_orient_err_type y_orient_err;
  using _z_orient_err_type =
    double;
  _z_orient_err_type z_orient_err;
  using _w_orient_err_type =
    double;
  _w_orient_err_type w_orient_err;
  using _map_similarity_type =
    double;
  _map_similarity_type map_similarity;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__x_err(
    const double & _arg)
  {
    this->x_err = _arg;
    return *this;
  }
  Type & set__y_err(
    const double & _arg)
  {
    this->y_err = _arg;
    return *this;
  }
  Type & set__z_err(
    const double & _arg)
  {
    this->z_err = _arg;
    return *this;
  }
  Type & set__x_orient_err(
    const double & _arg)
  {
    this->x_orient_err = _arg;
    return *this;
  }
  Type & set__y_orient_err(
    const double & _arg)
  {
    this->y_orient_err = _arg;
    return *this;
  }
  Type & set__z_orient_err(
    const double & _arg)
  {
    this->z_orient_err = _arg;
    return *this;
  }
  Type & set__w_orient_err(
    const double & _arg)
  {
    this->w_orient_err = _arg;
    return *this;
  }
  Type & set__map_similarity(
    const double & _arg)
  {
    this->map_similarity = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::SLAMErr_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::SLAMErr_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::SLAMErr_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::SLAMErr_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::SLAMErr_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::SLAMErr_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::SLAMErr_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::SLAMErr_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::SLAMErr_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::SLAMErr_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__SLAMErr
    std::shared_ptr<eufs_msgs::msg::SLAMErr_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__SLAMErr
    std::shared_ptr<eufs_msgs::msg::SLAMErr_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SLAMErr_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->x_err != other.x_err) {
      return false;
    }
    if (this->y_err != other.y_err) {
      return false;
    }
    if (this->z_err != other.z_err) {
      return false;
    }
    if (this->x_orient_err != other.x_orient_err) {
      return false;
    }
    if (this->y_orient_err != other.y_orient_err) {
      return false;
    }
    if (this->z_orient_err != other.z_orient_err) {
      return false;
    }
    if (this->w_orient_err != other.w_orient_err) {
      return false;
    }
    if (this->map_similarity != other.map_similarity) {
      return false;
    }
    return true;
  }
  bool operator!=(const SLAMErr_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SLAMErr_

// alias to use template instance with default allocator
using SLAMErr =
  eufs_msgs::msg::SLAMErr_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__SLAM_ERR__STRUCT_HPP_
