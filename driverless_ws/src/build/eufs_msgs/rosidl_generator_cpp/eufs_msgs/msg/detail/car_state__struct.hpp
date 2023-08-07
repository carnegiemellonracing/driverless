// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/CarState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CAR_STATE__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__CAR_STATE__STRUCT_HPP_

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
// Member 'pose'
#include "geometry_msgs/msg/detail/pose_with_covariance__struct.hpp"
// Member 'twist'
#include "geometry_msgs/msg/detail/twist_with_covariance__struct.hpp"
// Member 'linear_acceleration'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__CarState __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__CarState __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct CarState_
{
  using Type = CarState_<ContainerAllocator>;

  explicit CarState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    pose(_init),
    twist(_init),
    linear_acceleration(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->child_frame_id = "";
      std::fill<typename std::array<double, 9>::iterator, double>(this->linear_acceleration_covariance.begin(), this->linear_acceleration_covariance.end(), 0.0);
      this->slip_angle = 0.0;
      this->state_of_charge = 0.0;
    }
  }

  explicit CarState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    child_frame_id(_alloc),
    pose(_alloc, _init),
    twist(_alloc, _init),
    linear_acceleration(_alloc, _init),
    linear_acceleration_covariance(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->child_frame_id = "";
      std::fill<typename std::array<double, 9>::iterator, double>(this->linear_acceleration_covariance.begin(), this->linear_acceleration_covariance.end(), 0.0);
      this->slip_angle = 0.0;
      this->state_of_charge = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _child_frame_id_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _child_frame_id_type child_frame_id;
  using _pose_type =
    geometry_msgs::msg::PoseWithCovariance_<ContainerAllocator>;
  _pose_type pose;
  using _twist_type =
    geometry_msgs::msg::TwistWithCovariance_<ContainerAllocator>;
  _twist_type twist;
  using _linear_acceleration_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _linear_acceleration_type linear_acceleration;
  using _linear_acceleration_covariance_type =
    std::array<double, 9>;
  _linear_acceleration_covariance_type linear_acceleration_covariance;
  using _slip_angle_type =
    double;
  _slip_angle_type slip_angle;
  using _state_of_charge_type =
    double;
  _state_of_charge_type state_of_charge;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__child_frame_id(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->child_frame_id = _arg;
    return *this;
  }
  Type & set__pose(
    const geometry_msgs::msg::PoseWithCovariance_<ContainerAllocator> & _arg)
  {
    this->pose = _arg;
    return *this;
  }
  Type & set__twist(
    const geometry_msgs::msg::TwistWithCovariance_<ContainerAllocator> & _arg)
  {
    this->twist = _arg;
    return *this;
  }
  Type & set__linear_acceleration(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->linear_acceleration = _arg;
    return *this;
  }
  Type & set__linear_acceleration_covariance(
    const std::array<double, 9> & _arg)
  {
    this->linear_acceleration_covariance = _arg;
    return *this;
  }
  Type & set__slip_angle(
    const double & _arg)
  {
    this->slip_angle = _arg;
    return *this;
  }
  Type & set__state_of_charge(
    const double & _arg)
  {
    this->state_of_charge = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::CarState_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::CarState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::CarState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::CarState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::CarState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::CarState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::CarState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::CarState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::CarState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::CarState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__CarState
    std::shared_ptr<eufs_msgs::msg::CarState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__CarState
    std::shared_ptr<eufs_msgs::msg::CarState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CarState_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->child_frame_id != other.child_frame_id) {
      return false;
    }
    if (this->pose != other.pose) {
      return false;
    }
    if (this->twist != other.twist) {
      return false;
    }
    if (this->linear_acceleration != other.linear_acceleration) {
      return false;
    }
    if (this->linear_acceleration_covariance != other.linear_acceleration_covariance) {
      return false;
    }
    if (this->slip_angle != other.slip_angle) {
      return false;
    }
    if (this->state_of_charge != other.state_of_charge) {
      return false;
    }
    return true;
  }
  bool operator!=(const CarState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CarState_

// alias to use template instance with default allocator
using CarState =
  eufs_msgs::msg::CarState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__CAR_STATE__STRUCT_HPP_
