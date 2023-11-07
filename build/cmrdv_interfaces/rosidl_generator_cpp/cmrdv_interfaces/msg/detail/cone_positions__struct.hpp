// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cmrdv_interfaces:msg/ConePositions.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__STRUCT_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__STRUCT_HPP_

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
// Member 'cone_positions'
#include "std_msgs/msg/detail/float32__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__cmrdv_interfaces__msg__ConePositions __attribute__((deprecated))
#else
# define DEPRECATED__cmrdv_interfaces__msg__ConePositions __declspec(deprecated)
#endif

namespace cmrdv_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ConePositions_
{
  using Type = ConePositions_<ContainerAllocator>;

  explicit ConePositions_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    (void)_init;
  }

  explicit ConePositions_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _cone_positions_type =
    std::vector<std_msgs::msg::Float32_<ContainerAllocator>, typename ContainerAllocator::template rebind<std_msgs::msg::Float32_<ContainerAllocator>>::other>;
  _cone_positions_type cone_positions;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__cone_positions(
    const std::vector<std_msgs::msg::Float32_<ContainerAllocator>, typename ContainerAllocator::template rebind<std_msgs::msg::Float32_<ContainerAllocator>>::other> & _arg)
  {
    this->cone_positions = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cmrdv_interfaces::msg::ConePositions_<ContainerAllocator> *;
  using ConstRawPtr =
    const cmrdv_interfaces::msg::ConePositions_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::ConePositions_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::ConePositions_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::ConePositions_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::ConePositions_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::ConePositions_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::ConePositions_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::ConePositions_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::ConePositions_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cmrdv_interfaces__msg__ConePositions
    std::shared_ptr<cmrdv_interfaces::msg::ConePositions_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cmrdv_interfaces__msg__ConePositions
    std::shared_ptr<cmrdv_interfaces::msg::ConePositions_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ConePositions_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->cone_positions != other.cone_positions) {
      return false;
    }
    return true;
  }
  bool operator!=(const ConePositions_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ConePositions_

// alias to use template instance with default allocator
using ConePositions =
  cmrdv_interfaces::msg::ConePositions_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__STRUCT_HPP_
