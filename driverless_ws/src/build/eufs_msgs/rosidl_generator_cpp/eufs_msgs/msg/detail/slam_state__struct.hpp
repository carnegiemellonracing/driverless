// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/SLAMState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__SLAM_STATE__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__SLAM_STATE__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__SLAMState __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__SLAMState __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SLAMState_
{
  using Type = SLAMState_<ContainerAllocator>;

  explicit SLAMState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->loop_closed = false;
      this->laps = 0;
      this->status = "";
      this->state = 0;
    }
  }

  explicit SLAMState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : status(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->loop_closed = false;
      this->laps = 0;
      this->status = "";
      this->state = 0;
    }
  }

  // field types and members
  using _loop_closed_type =
    bool;
  _loop_closed_type loop_closed;
  using _laps_type =
    int16_t;
  _laps_type laps;
  using _status_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _status_type status;
  using _state_type =
    int8_t;
  _state_type state;

  // setters for named parameter idiom
  Type & set__loop_closed(
    const bool & _arg)
  {
    this->loop_closed = _arg;
    return *this;
  }
  Type & set__laps(
    const int16_t & _arg)
  {
    this->laps = _arg;
    return *this;
  }
  Type & set__status(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->status = _arg;
    return *this;
  }
  Type & set__state(
    const int8_t & _arg)
  {
    this->state = _arg;
    return *this;
  }

  // constant declarations
  static constexpr int8_t NO_INPUTS =
    0;
  static constexpr int8_t MAPPING =
    1;
  static constexpr int8_t LOCALISING =
    2;
  static constexpr int8_t TIMEOUT =
    3;
  static constexpr int8_t RECOMMENDS_FAILURE =
    4;

  // pointer types
  using RawPtr =
    eufs_msgs::msg::SLAMState_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::SLAMState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::SLAMState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::SLAMState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::SLAMState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::SLAMState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::SLAMState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::SLAMState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::SLAMState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::SLAMState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__SLAMState
    std::shared_ptr<eufs_msgs::msg::SLAMState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__SLAMState
    std::shared_ptr<eufs_msgs::msg::SLAMState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SLAMState_ & other) const
  {
    if (this->loop_closed != other.loop_closed) {
      return false;
    }
    if (this->laps != other.laps) {
      return false;
    }
    if (this->status != other.status) {
      return false;
    }
    if (this->state != other.state) {
      return false;
    }
    return true;
  }
  bool operator!=(const SLAMState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SLAMState_

// alias to use template instance with default allocator
using SLAMState =
  eufs_msgs::msg::SLAMState_<std::allocator<void>>;

// constant definitions
template<typename ContainerAllocator>
constexpr int8_t SLAMState_<ContainerAllocator>::NO_INPUTS;
template<typename ContainerAllocator>
constexpr int8_t SLAMState_<ContainerAllocator>::MAPPING;
template<typename ContainerAllocator>
constexpr int8_t SLAMState_<ContainerAllocator>::LOCALISING;
template<typename ContainerAllocator>
constexpr int8_t SLAMState_<ContainerAllocator>::TIMEOUT;
template<typename ContainerAllocator>
constexpr int8_t SLAMState_<ContainerAllocator>::RECOMMENDS_FAILURE;

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__SLAM_STATE__STRUCT_HPP_
