// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cmrdv_interfaces:msg/ControlAction.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__STRUCT_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__cmrdv_interfaces__msg__ControlAction __attribute__((deprecated))
#else
# define DEPRECATED__cmrdv_interfaces__msg__ControlAction __declspec(deprecated)
#endif

namespace cmrdv_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ControlAction_
{
  using Type = ControlAction_<ContainerAllocator>;

  explicit ControlAction_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->wheel_speed = 0.0;
      this->swangle = 0.0;
    }
  }

  explicit ControlAction_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->wheel_speed = 0.0;
      this->swangle = 0.0;
    }
  }

  // field types and members
  using _wheel_speed_type =
    double;
  _wheel_speed_type wheel_speed;
  using _swangle_type =
    double;
  _swangle_type swangle;

  // setters for named parameter idiom
  Type & set__wheel_speed(
    const double & _arg)
  {
    this->wheel_speed = _arg;
    return *this;
  }
  Type & set__swangle(
    const double & _arg)
  {
    this->swangle = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cmrdv_interfaces::msg::ControlAction_<ContainerAllocator> *;
  using ConstRawPtr =
    const cmrdv_interfaces::msg::ControlAction_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::ControlAction_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::ControlAction_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::ControlAction_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::ControlAction_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::ControlAction_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::ControlAction_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::ControlAction_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::ControlAction_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cmrdv_interfaces__msg__ControlAction
    std::shared_ptr<cmrdv_interfaces::msg::ControlAction_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cmrdv_interfaces__msg__ControlAction
    std::shared_ptr<cmrdv_interfaces::msg::ControlAction_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ControlAction_ & other) const
  {
    if (this->wheel_speed != other.wheel_speed) {
      return false;
    }
    if (this->swangle != other.swangle) {
      return false;
    }
    return true;
  }
  bool operator!=(const ControlAction_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ControlAction_

// alias to use template instance with default allocator
using ControlAction =
  cmrdv_interfaces::msg::ControlAction_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__STRUCT_HPP_
