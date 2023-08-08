// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/MPCState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__MPC_STATE__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__MPC_STATE__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__MPCState __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__MPCState __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MPCState_
{
  using Type = MPCState_<ContainerAllocator>;

  explicit MPCState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->exitflag = 0;
      this->iterations = 0;
      this->solve_time = 0.0;
      this->cost = 0.0;
    }
  }

  explicit MPCState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->exitflag = 0;
      this->iterations = 0;
      this->solve_time = 0.0;
      this->cost = 0.0;
    }
  }

  // field types and members
  using _exitflag_type =
    int8_t;
  _exitflag_type exitflag;
  using _iterations_type =
    uint8_t;
  _iterations_type iterations;
  using _solve_time_type =
    double;
  _solve_time_type solve_time;
  using _cost_type =
    double;
  _cost_type cost;

  // setters for named parameter idiom
  Type & set__exitflag(
    const int8_t & _arg)
  {
    this->exitflag = _arg;
    return *this;
  }
  Type & set__iterations(
    const uint8_t & _arg)
  {
    this->iterations = _arg;
    return *this;
  }
  Type & set__solve_time(
    const double & _arg)
  {
    this->solve_time = _arg;
    return *this;
  }
  Type & set__cost(
    const double & _arg)
  {
    this->cost = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::MPCState_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::MPCState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::MPCState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::MPCState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::MPCState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::MPCState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::MPCState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::MPCState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::MPCState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::MPCState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__MPCState
    std::shared_ptr<eufs_msgs::msg::MPCState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__MPCState
    std::shared_ptr<eufs_msgs::msg::MPCState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MPCState_ & other) const
  {
    if (this->exitflag != other.exitflag) {
      return false;
    }
    if (this->iterations != other.iterations) {
      return false;
    }
    if (this->solve_time != other.solve_time) {
      return false;
    }
    if (this->cost != other.cost) {
      return false;
    }
    return true;
  }
  bool operator!=(const MPCState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MPCState_

// alias to use template instance with default allocator
using MPCState =
  eufs_msgs::msg::MPCState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__MPC_STATE__STRUCT_HPP_
