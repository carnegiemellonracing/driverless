// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/WheelSpeeds.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__WheelSpeeds __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__WheelSpeeds __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct WheelSpeeds_
{
  using Type = WheelSpeeds_<ContainerAllocator>;

  explicit WheelSpeeds_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->steering = 0.0f;
      this->lf_speed = 0.0f;
      this->rf_speed = 0.0f;
      this->lb_speed = 0.0f;
      this->rb_speed = 0.0f;
    }
  }

  explicit WheelSpeeds_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->steering = 0.0f;
      this->lf_speed = 0.0f;
      this->rf_speed = 0.0f;
      this->lb_speed = 0.0f;
      this->rb_speed = 0.0f;
    }
  }

  // field types and members
  using _steering_type =
    float;
  _steering_type steering;
  using _lf_speed_type =
    float;
  _lf_speed_type lf_speed;
  using _rf_speed_type =
    float;
  _rf_speed_type rf_speed;
  using _lb_speed_type =
    float;
  _lb_speed_type lb_speed;
  using _rb_speed_type =
    float;
  _rb_speed_type rb_speed;

  // setters for named parameter idiom
  Type & set__steering(
    const float & _arg)
  {
    this->steering = _arg;
    return *this;
  }
  Type & set__lf_speed(
    const float & _arg)
  {
    this->lf_speed = _arg;
    return *this;
  }
  Type & set__rf_speed(
    const float & _arg)
  {
    this->rf_speed = _arg;
    return *this;
  }
  Type & set__lb_speed(
    const float & _arg)
  {
    this->lb_speed = _arg;
    return *this;
  }
  Type & set__rb_speed(
    const float & _arg)
  {
    this->rb_speed = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::WheelSpeeds_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::WheelSpeeds_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::WheelSpeeds_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::WheelSpeeds_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::WheelSpeeds_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::WheelSpeeds_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::WheelSpeeds_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::WheelSpeeds_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::WheelSpeeds_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::WheelSpeeds_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__WheelSpeeds
    std::shared_ptr<eufs_msgs::msg::WheelSpeeds_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__WheelSpeeds
    std::shared_ptr<eufs_msgs::msg::WheelSpeeds_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const WheelSpeeds_ & other) const
  {
    if (this->steering != other.steering) {
      return false;
    }
    if (this->lf_speed != other.lf_speed) {
      return false;
    }
    if (this->rf_speed != other.rf_speed) {
      return false;
    }
    if (this->lb_speed != other.lb_speed) {
      return false;
    }
    if (this->rb_speed != other.rb_speed) {
      return false;
    }
    return true;
  }
  bool operator!=(const WheelSpeeds_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct WheelSpeeds_

// alias to use template instance with default allocator
using WheelSpeeds =
  eufs_msgs::msg::WheelSpeeds_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__STRUCT_HPP_
