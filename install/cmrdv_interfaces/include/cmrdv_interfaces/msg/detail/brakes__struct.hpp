// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cmrdv_interfaces:msg/Brakes.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__BRAKES__STRUCT_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__BRAKES__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'last_fired'
#include "builtin_interfaces/msg/detail/time__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__cmrdv_interfaces__msg__Brakes __attribute__((deprecated))
#else
# define DEPRECATED__cmrdv_interfaces__msg__Brakes __declspec(deprecated)
#endif

namespace cmrdv_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Brakes_
{
  using Type = Brakes_<ContainerAllocator>;

  explicit Brakes_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : last_fired(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->braking = false;
    }
  }

  explicit Brakes_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : last_fired(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->braking = false;
    }
  }

  // field types and members
  using _braking_type =
    bool;
  _braking_type braking;
  using _last_fired_type =
    builtin_interfaces::msg::Time_<ContainerAllocator>;
  _last_fired_type last_fired;

  // setters for named parameter idiom
  Type & set__braking(
    const bool & _arg)
  {
    this->braking = _arg;
    return *this;
  }
  Type & set__last_fired(
    const builtin_interfaces::msg::Time_<ContainerAllocator> & _arg)
  {
    this->last_fired = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cmrdv_interfaces::msg::Brakes_<ContainerAllocator> *;
  using ConstRawPtr =
    const cmrdv_interfaces::msg::Brakes_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::Brakes_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::Brakes_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::Brakes_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::Brakes_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::Brakes_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::Brakes_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::Brakes_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::Brakes_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cmrdv_interfaces__msg__Brakes
    std::shared_ptr<cmrdv_interfaces::msg::Brakes_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cmrdv_interfaces__msg__Brakes
    std::shared_ptr<cmrdv_interfaces::msg::Brakes_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Brakes_ & other) const
  {
    if (this->braking != other.braking) {
      return false;
    }
    if (this->last_fired != other.last_fired) {
      return false;
    }
    return true;
  }
  bool operator!=(const Brakes_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Brakes_

// alias to use template instance with default allocator
using Brakes =
  cmrdv_interfaces::msg::Brakes_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__BRAKES__STRUCT_HPP_
