// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/PathIntegralStats.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__STRUCT_HPP_

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
// Member 'params'
#include "eufs_msgs/msg/detail/path_integral_params__struct.hpp"
// Member 'stats'
#include "eufs_msgs/msg/detail/lap_stats__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__PathIntegralStats __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__PathIntegralStats __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PathIntegralStats_
{
  using Type = PathIntegralStats_<ContainerAllocator>;

  explicit PathIntegralStats_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    params(_init),
    stats(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->tag = "";
    }
  }

  explicit PathIntegralStats_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    tag(_alloc),
    params(_alloc, _init),
    stats(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->tag = "";
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _tag_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _tag_type tag;
  using _params_type =
    eufs_msgs::msg::PathIntegralParams_<ContainerAllocator>;
  _params_type params;
  using _stats_type =
    eufs_msgs::msg::LapStats_<ContainerAllocator>;
  _stats_type stats;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__tag(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->tag = _arg;
    return *this;
  }
  Type & set__params(
    const eufs_msgs::msg::PathIntegralParams_<ContainerAllocator> & _arg)
  {
    this->params = _arg;
    return *this;
  }
  Type & set__stats(
    const eufs_msgs::msg::LapStats_<ContainerAllocator> & _arg)
  {
    this->stats = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::PathIntegralStats_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::PathIntegralStats_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::PathIntegralStats_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::PathIntegralStats_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PathIntegralStats_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PathIntegralStats_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PathIntegralStats_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PathIntegralStats_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::PathIntegralStats_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::PathIntegralStats_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__PathIntegralStats
    std::shared_ptr<eufs_msgs::msg::PathIntegralStats_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__PathIntegralStats
    std::shared_ptr<eufs_msgs::msg::PathIntegralStats_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PathIntegralStats_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->tag != other.tag) {
      return false;
    }
    if (this->params != other.params) {
      return false;
    }
    if (this->stats != other.stats) {
      return false;
    }
    return true;
  }
  bool operator!=(const PathIntegralStats_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PathIntegralStats_

// alias to use template instance with default allocator
using PathIntegralStats =
  eufs_msgs::msg::PathIntegralStats_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__STRUCT_HPP_
