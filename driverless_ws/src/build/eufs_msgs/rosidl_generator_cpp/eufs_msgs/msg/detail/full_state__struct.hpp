// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/FullState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__FULL_STATE__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__FULL_STATE__STRUCT_HPP_

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
# define DEPRECATED__eufs_msgs__msg__FullState __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__FullState __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct FullState_
{
  using Type = FullState_<ContainerAllocator>;

  explicit FullState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x_pos = 0.0;
      this->y_pos = 0.0;
      this->yaw = 0.0;
      this->roll = 0.0;
      this->u_x = 0.0;
      this->u_y = 0.0;
      this->yaw_mder = 0.0;
      this->front_throttle = 0.0;
      this->rear_throttle = 0.0;
      this->steering = 0.0;
    }
  }

  explicit FullState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x_pos = 0.0;
      this->y_pos = 0.0;
      this->yaw = 0.0;
      this->roll = 0.0;
      this->u_x = 0.0;
      this->u_y = 0.0;
      this->yaw_mder = 0.0;
      this->front_throttle = 0.0;
      this->rear_throttle = 0.0;
      this->steering = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _x_pos_type =
    double;
  _x_pos_type x_pos;
  using _y_pos_type =
    double;
  _y_pos_type y_pos;
  using _yaw_type =
    double;
  _yaw_type yaw;
  using _roll_type =
    double;
  _roll_type roll;
  using _u_x_type =
    double;
  _u_x_type u_x;
  using _u_y_type =
    double;
  _u_y_type u_y;
  using _yaw_mder_type =
    double;
  _yaw_mder_type yaw_mder;
  using _front_throttle_type =
    double;
  _front_throttle_type front_throttle;
  using _rear_throttle_type =
    double;
  _rear_throttle_type rear_throttle;
  using _steering_type =
    double;
  _steering_type steering;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__x_pos(
    const double & _arg)
  {
    this->x_pos = _arg;
    return *this;
  }
  Type & set__y_pos(
    const double & _arg)
  {
    this->y_pos = _arg;
    return *this;
  }
  Type & set__yaw(
    const double & _arg)
  {
    this->yaw = _arg;
    return *this;
  }
  Type & set__roll(
    const double & _arg)
  {
    this->roll = _arg;
    return *this;
  }
  Type & set__u_x(
    const double & _arg)
  {
    this->u_x = _arg;
    return *this;
  }
  Type & set__u_y(
    const double & _arg)
  {
    this->u_y = _arg;
    return *this;
  }
  Type & set__yaw_mder(
    const double & _arg)
  {
    this->yaw_mder = _arg;
    return *this;
  }
  Type & set__front_throttle(
    const double & _arg)
  {
    this->front_throttle = _arg;
    return *this;
  }
  Type & set__rear_throttle(
    const double & _arg)
  {
    this->rear_throttle = _arg;
    return *this;
  }
  Type & set__steering(
    const double & _arg)
  {
    this->steering = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::FullState_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::FullState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::FullState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::FullState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::FullState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::FullState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::FullState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::FullState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::FullState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::FullState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__FullState
    std::shared_ptr<eufs_msgs::msg::FullState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__FullState
    std::shared_ptr<eufs_msgs::msg::FullState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const FullState_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->x_pos != other.x_pos) {
      return false;
    }
    if (this->y_pos != other.y_pos) {
      return false;
    }
    if (this->yaw != other.yaw) {
      return false;
    }
    if (this->roll != other.roll) {
      return false;
    }
    if (this->u_x != other.u_x) {
      return false;
    }
    if (this->u_y != other.u_y) {
      return false;
    }
    if (this->yaw_mder != other.yaw_mder) {
      return false;
    }
    if (this->front_throttle != other.front_throttle) {
      return false;
    }
    if (this->rear_throttle != other.rear_throttle) {
      return false;
    }
    if (this->steering != other.steering) {
      return false;
    }
    return true;
  }
  bool operator!=(const FullState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct FullState_

// alias to use template instance with default allocator
using FullState =
  eufs_msgs::msg::FullState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__FULL_STATE__STRUCT_HPP_
