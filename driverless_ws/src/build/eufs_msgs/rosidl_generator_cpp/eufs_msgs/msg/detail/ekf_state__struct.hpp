// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/EKFState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__EKF_STATE__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__EKF_STATE__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__EKFState __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__EKFState __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct EKFState_
{
  using Type = EKFState_<ContainerAllocator>;

  explicit EKFState_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->gps_received = false;
      this->imu_received = false;
      this->wheel_odom_received = false;
      this->ekf_odom_received = false;
      this->ekf_accel_received = false;
      this->currently_over_covariance_limit = false;
      this->consecutive_turns_over_covariance_limit = false;
      this->recommends_failure = false;
    }
  }

  explicit EKFState_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->gps_received = false;
      this->imu_received = false;
      this->wheel_odom_received = false;
      this->ekf_odom_received = false;
      this->ekf_accel_received = false;
      this->currently_over_covariance_limit = false;
      this->consecutive_turns_over_covariance_limit = false;
      this->recommends_failure = false;
    }
  }

  // field types and members
  using _gps_received_type =
    bool;
  _gps_received_type gps_received;
  using _imu_received_type =
    bool;
  _imu_received_type imu_received;
  using _wheel_odom_received_type =
    bool;
  _wheel_odom_received_type wheel_odom_received;
  using _ekf_odom_received_type =
    bool;
  _ekf_odom_received_type ekf_odom_received;
  using _ekf_accel_received_type =
    bool;
  _ekf_accel_received_type ekf_accel_received;
  using _currently_over_covariance_limit_type =
    bool;
  _currently_over_covariance_limit_type currently_over_covariance_limit;
  using _consecutive_turns_over_covariance_limit_type =
    bool;
  _consecutive_turns_over_covariance_limit_type consecutive_turns_over_covariance_limit;
  using _recommends_failure_type =
    bool;
  _recommends_failure_type recommends_failure;

  // setters for named parameter idiom
  Type & set__gps_received(
    const bool & _arg)
  {
    this->gps_received = _arg;
    return *this;
  }
  Type & set__imu_received(
    const bool & _arg)
  {
    this->imu_received = _arg;
    return *this;
  }
  Type & set__wheel_odom_received(
    const bool & _arg)
  {
    this->wheel_odom_received = _arg;
    return *this;
  }
  Type & set__ekf_odom_received(
    const bool & _arg)
  {
    this->ekf_odom_received = _arg;
    return *this;
  }
  Type & set__ekf_accel_received(
    const bool & _arg)
  {
    this->ekf_accel_received = _arg;
    return *this;
  }
  Type & set__currently_over_covariance_limit(
    const bool & _arg)
  {
    this->currently_over_covariance_limit = _arg;
    return *this;
  }
  Type & set__consecutive_turns_over_covariance_limit(
    const bool & _arg)
  {
    this->consecutive_turns_over_covariance_limit = _arg;
    return *this;
  }
  Type & set__recommends_failure(
    const bool & _arg)
  {
    this->recommends_failure = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::EKFState_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::EKFState_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::EKFState_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::EKFState_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::EKFState_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::EKFState_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::EKFState_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::EKFState_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::EKFState_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::EKFState_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__EKFState
    std::shared_ptr<eufs_msgs::msg::EKFState_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__EKFState
    std::shared_ptr<eufs_msgs::msg::EKFState_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const EKFState_ & other) const
  {
    if (this->gps_received != other.gps_received) {
      return false;
    }
    if (this->imu_received != other.imu_received) {
      return false;
    }
    if (this->wheel_odom_received != other.wheel_odom_received) {
      return false;
    }
    if (this->ekf_odom_received != other.ekf_odom_received) {
      return false;
    }
    if (this->ekf_accel_received != other.ekf_accel_received) {
      return false;
    }
    if (this->currently_over_covariance_limit != other.currently_over_covariance_limit) {
      return false;
    }
    if (this->consecutive_turns_over_covariance_limit != other.consecutive_turns_over_covariance_limit) {
      return false;
    }
    if (this->recommends_failure != other.recommends_failure) {
      return false;
    }
    return true;
  }
  bool operator!=(const EKFState_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct EKFState_

// alias to use template instance with default allocator
using EKFState =
  eufs_msgs::msg::EKFState_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__EKF_STATE__STRUCT_HPP_
