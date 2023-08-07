// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/VehicleCommands.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__VehicleCommands __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__VehicleCommands __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct VehicleCommands_
{
  using Type = VehicleCommands_<ContainerAllocator>;

  explicit VehicleCommands_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->handshake = 0;
      this->ebs = 0;
      this->direction = 0;
      this->mission_status = 0;
      this->braking = 0.0;
      this->torque = 0.0;
      this->steering = 0.0;
      this->rpm = 0.0;
    }
  }

  explicit VehicleCommands_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->handshake = 0;
      this->ebs = 0;
      this->direction = 0;
      this->mission_status = 0;
      this->braking = 0.0;
      this->torque = 0.0;
      this->steering = 0.0;
      this->rpm = 0.0;
    }
  }

  // field types and members
  using _handshake_type =
    int8_t;
  _handshake_type handshake;
  using _ebs_type =
    int8_t;
  _ebs_type ebs;
  using _direction_type =
    int8_t;
  _direction_type direction;
  using _mission_status_type =
    int8_t;
  _mission_status_type mission_status;
  using _braking_type =
    double;
  _braking_type braking;
  using _torque_type =
    double;
  _torque_type torque;
  using _steering_type =
    double;
  _steering_type steering;
  using _rpm_type =
    double;
  _rpm_type rpm;

  // setters for named parameter idiom
  Type & set__handshake(
    const int8_t & _arg)
  {
    this->handshake = _arg;
    return *this;
  }
  Type & set__ebs(
    const int8_t & _arg)
  {
    this->ebs = _arg;
    return *this;
  }
  Type & set__direction(
    const int8_t & _arg)
  {
    this->direction = _arg;
    return *this;
  }
  Type & set__mission_status(
    const int8_t & _arg)
  {
    this->mission_status = _arg;
    return *this;
  }
  Type & set__braking(
    const double & _arg)
  {
    this->braking = _arg;
    return *this;
  }
  Type & set__torque(
    const double & _arg)
  {
    this->torque = _arg;
    return *this;
  }
  Type & set__steering(
    const double & _arg)
  {
    this->steering = _arg;
    return *this;
  }
  Type & set__rpm(
    const double & _arg)
  {
    this->rpm = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::VehicleCommands_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::VehicleCommands_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::VehicleCommands_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::VehicleCommands_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::VehicleCommands_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::VehicleCommands_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::VehicleCommands_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::VehicleCommands_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::VehicleCommands_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::VehicleCommands_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__VehicleCommands
    std::shared_ptr<eufs_msgs::msg::VehicleCommands_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__VehicleCommands
    std::shared_ptr<eufs_msgs::msg::VehicleCommands_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const VehicleCommands_ & other) const
  {
    if (this->handshake != other.handshake) {
      return false;
    }
    if (this->ebs != other.ebs) {
      return false;
    }
    if (this->direction != other.direction) {
      return false;
    }
    if (this->mission_status != other.mission_status) {
      return false;
    }
    if (this->braking != other.braking) {
      return false;
    }
    if (this->torque != other.torque) {
      return false;
    }
    if (this->steering != other.steering) {
      return false;
    }
    if (this->rpm != other.rpm) {
      return false;
    }
    return true;
  }
  bool operator!=(const VehicleCommands_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct VehicleCommands_

// alias to use template instance with default allocator
using VehicleCommands =
  eufs_msgs::msg::VehicleCommands_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__STRUCT_HPP_
