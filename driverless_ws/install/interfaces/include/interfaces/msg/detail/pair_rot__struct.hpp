// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from interfaces:msg/PairROT.idl
// generated code does not contain a copyright notice

#ifndef INTERFACES__MSG__DETAIL__PAIR_ROT__STRUCT_HPP_
#define INTERFACES__MSG__DETAIL__PAIR_ROT__STRUCT_HPP_

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
// Member 'near'
// Member 'far'
#include "interfaces/msg/detail/car_rot__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__interfaces__msg__PairROT __attribute__((deprecated))
#else
# define DEPRECATED__interfaces__msg__PairROT __declspec(deprecated)
#endif

namespace interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PairROT_
{
  using Type = PairROT_<ContainerAllocator>;

  explicit PairROT_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    near(_init),
    far(_init)
  {
    (void)_init;
  }

  explicit PairROT_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    near(_alloc, _init),
    far(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _near_type =
    interfaces::msg::CarROT_<ContainerAllocator>;
  _near_type near;
  using _far_type =
    interfaces::msg::CarROT_<ContainerAllocator>;
  _far_type far;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__near(
    const interfaces::msg::CarROT_<ContainerAllocator> & _arg)
  {
    this->near = _arg;
    return *this;
  }
  Type & set__far(
    const interfaces::msg::CarROT_<ContainerAllocator> & _arg)
  {
    this->far = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    interfaces::msg::PairROT_<ContainerAllocator> *;
  using ConstRawPtr =
    const interfaces::msg::PairROT_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<interfaces::msg::PairROT_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<interfaces::msg::PairROT_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      interfaces::msg::PairROT_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<interfaces::msg::PairROT_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      interfaces::msg::PairROT_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<interfaces::msg::PairROT_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<interfaces::msg::PairROT_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<interfaces::msg::PairROT_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__interfaces__msg__PairROT
    std::shared_ptr<interfaces::msg::PairROT_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__interfaces__msg__PairROT
    std::shared_ptr<interfaces::msg::PairROT_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PairROT_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->near != other.near) {
      return false;
    }
    if (this->far != other.far) {
      return false;
    }
    return true;
  }
  bool operator!=(const PairROT_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PairROT_

// alias to use template instance with default allocator
using PairROT =
  interfaces::msg::PairROT_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace interfaces

#endif  // INTERFACES__MSG__DETAIL__PAIR_ROT__STRUCT_HPP_
