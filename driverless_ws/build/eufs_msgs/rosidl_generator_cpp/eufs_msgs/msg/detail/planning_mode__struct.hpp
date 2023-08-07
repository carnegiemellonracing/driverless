// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/PlanningMode.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__PlanningMode __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__PlanningMode __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PlanningMode_
{
  using Type = PlanningMode_<ContainerAllocator>;

  explicit PlanningMode_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mode = 0;
    }
  }

  explicit PlanningMode_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->mode = 0;
    }
  }

  // field types and members
  using _mode_type =
    int8_t;
  _mode_type mode;

  // setters for named parameter idiom
  Type & set__mode(
    const int8_t & _arg)
  {
    this->mode = _arg;
    return *this;
  }

  // constant declarations
  static constexpr int8_t LOCAL =
    0;
  static constexpr int8_t GLOBAL =
    1;

  // pointer types
  using RawPtr =
    eufs_msgs::msg::PlanningMode_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::PlanningMode_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::PlanningMode_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::PlanningMode_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PlanningMode_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PlanningMode_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PlanningMode_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PlanningMode_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::PlanningMode_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::PlanningMode_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__PlanningMode
    std::shared_ptr<eufs_msgs::msg::PlanningMode_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__PlanningMode
    std::shared_ptr<eufs_msgs::msg::PlanningMode_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PlanningMode_ & other) const
  {
    if (this->mode != other.mode) {
      return false;
    }
    return true;
  }
  bool operator!=(const PlanningMode_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PlanningMode_

// alias to use template instance with default allocator
using PlanningMode =
  eufs_msgs::msg::PlanningMode_<std::allocator<void>>;

// constant definitions
template<typename ContainerAllocator>
constexpr int8_t PlanningMode_<ContainerAllocator>::LOCAL;
template<typename ContainerAllocator>
constexpr int8_t PlanningMode_<ContainerAllocator>::GLOBAL;

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__STRUCT_HPP_
