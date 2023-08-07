// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/CanState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CAN_STATE__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__CAN_STATE__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__CanState __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__CanState __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct CanState_
{
  using Type = CanState_<ContainerAllocator>;

  explicit CanState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->as_state = 0;
      this->ami_state = 0;
    }
  }

  explicit CanState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->as_state = 0;
      this->ami_state = 0;
    }
  }

  // field types and members
  using _as_state_type =
    uint16_t;
  _as_state_type as_state;
  using _ami_state_type =
    uint16_t;
  _ami_state_type ami_state;

  // setters for named parameter idiom
  Type & set__as_state(
    const uint16_t & _arg)
  {
    this->as_state = _arg;
    return *this;
  }
  Type & set__ami_state(
    const uint16_t & _arg)
  {
    this->ami_state = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint16_t AS_OFF =
    0u;
  static constexpr uint16_t AS_READY =
    1u;
  static constexpr uint16_t AS_DRIVING =
    2u;
  static constexpr uint16_t AS_EMERGENCY_BRAKE =
    3u;
  static constexpr uint16_t AS_FINISHED =
    4u;
  static constexpr uint16_t AMI_NOT_SELECTED =
    10u;
  static constexpr uint16_t AMI_ACCELERATION =
    11u;
  static constexpr uint16_t AMI_SKIDPAD =
    12u;
  static constexpr uint16_t AMI_AUTOCROSS =
    13u;
  static constexpr uint16_t AMI_TRACK_DRIVE =
    14u;
  static constexpr uint16_t AMI_AUTONOMOUS_DEMO =
    15u;
  static constexpr uint16_t AMI_ADS_INSPECTION =
    16u;
  static constexpr uint16_t AMI_ADS_EBS =
    17u;
  static constexpr uint16_t AMI_DDT_INSPECTION_A =
    18u;
  static constexpr uint16_t AMI_DDT_INSPECTION_B =
    19u;
  static constexpr uint16_t AMI_JOYSTICK =
    20u;
  static constexpr uint16_t AMI_MANUAL =
    21u;

  // pointer types
  using RawPtr =
    eufs_msgs::msg::CanState_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::CanState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::CanState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::CanState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::CanState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::CanState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::CanState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::CanState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::CanState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::CanState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__CanState
    std::shared_ptr<eufs_msgs::msg::CanState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__CanState
    std::shared_ptr<eufs_msgs::msg::CanState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CanState_ & other) const
  {
    if (this->as_state != other.as_state) {
      return false;
    }
    if (this->ami_state != other.ami_state) {
      return false;
    }
    return true;
  }
  bool operator!=(const CanState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CanState_

// alias to use template instance with default allocator
using CanState =
  eufs_msgs::msg::CanState_<std::allocator<void>>;

// constant definitions
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AS_OFF;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AS_READY;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AS_DRIVING;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AS_EMERGENCY_BRAKE;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AS_FINISHED;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_NOT_SELECTED;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_ACCELERATION;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_SKIDPAD;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_AUTOCROSS;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_TRACK_DRIVE;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_AUTONOMOUS_DEMO;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_ADS_INSPECTION;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_ADS_EBS;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_DDT_INSPECTION_A;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_DDT_INSPECTION_B;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_JOYSTICK;
template<typename ContainerAllocator>
constexpr uint16_t CanState_<ContainerAllocator>::AMI_MANUAL;

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__CAN_STATE__STRUCT_HPP_
