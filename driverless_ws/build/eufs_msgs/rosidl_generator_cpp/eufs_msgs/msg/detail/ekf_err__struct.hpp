// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/EKFErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__EKF_ERR__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__EKF_ERR__STRUCT_HPP_

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

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__EKFErr __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__EKFErr __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct EKFErr_
{
  using Type = EKFErr_<ContainerAllocator>;

  explicit EKFErr_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->gps_x_vel_err = 0.0;
      this->gps_y_vel_err = 0.0;
      this->imu_x_acc_err = 0.0;
      this->imu_y_acc_err = 0.0;
      this->imu_yaw_err = 0.0;
      this->ekf_x_vel_var = 0.0;
      this->ekf_y_vel_var = 0.0;
      this->ekf_x_acc_var = 0.0;
      this->ekf_y_acc_var = 0.0;
      this->ekf_yaw_var = 0.0;
    }
  }

  explicit EKFErr_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->gps_x_vel_err = 0.0;
      this->gps_y_vel_err = 0.0;
      this->imu_x_acc_err = 0.0;
      this->imu_y_acc_err = 0.0;
      this->imu_yaw_err = 0.0;
      this->ekf_x_vel_var = 0.0;
      this->ekf_y_vel_var = 0.0;
      this->ekf_x_acc_var = 0.0;
      this->ekf_y_acc_var = 0.0;
      this->ekf_yaw_var = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _gps_x_vel_err_type =
    double;
  _gps_x_vel_err_type gps_x_vel_err;
  using _gps_y_vel_err_type =
    double;
  _gps_y_vel_err_type gps_y_vel_err;
  using _imu_x_acc_err_type =
    double;
  _imu_x_acc_err_type imu_x_acc_err;
  using _imu_y_acc_err_type =
    double;
  _imu_y_acc_err_type imu_y_acc_err;
  using _imu_yaw_err_type =
    double;
  _imu_yaw_err_type imu_yaw_err;
  using _ekf_x_vel_var_type =
    double;
  _ekf_x_vel_var_type ekf_x_vel_var;
  using _ekf_y_vel_var_type =
    double;
  _ekf_y_vel_var_type ekf_y_vel_var;
  using _ekf_x_acc_var_type =
    double;
  _ekf_x_acc_var_type ekf_x_acc_var;
  using _ekf_y_acc_var_type =
    double;
  _ekf_y_acc_var_type ekf_y_acc_var;
  using _ekf_yaw_var_type =
    double;
  _ekf_yaw_var_type ekf_yaw_var;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__gps_x_vel_err(
    const double & _arg)
  {
    this->gps_x_vel_err = _arg;
    return *this;
  }
  Type & set__gps_y_vel_err(
    const double & _arg)
  {
    this->gps_y_vel_err = _arg;
    return *this;
  }
  Type & set__imu_x_acc_err(
    const double & _arg)
  {
    this->imu_x_acc_err = _arg;
    return *this;
  }
  Type & set__imu_y_acc_err(
    const double & _arg)
  {
    this->imu_y_acc_err = _arg;
    return *this;
  }
  Type & set__imu_yaw_err(
    const double & _arg)
  {
    this->imu_yaw_err = _arg;
    return *this;
  }
  Type & set__ekf_x_vel_var(
    const double & _arg)
  {
    this->ekf_x_vel_var = _arg;
    return *this;
  }
  Type & set__ekf_y_vel_var(
    const double & _arg)
  {
    this->ekf_y_vel_var = _arg;
    return *this;
  }
  Type & set__ekf_x_acc_var(
    const double & _arg)
  {
    this->ekf_x_acc_var = _arg;
    return *this;
  }
  Type & set__ekf_y_acc_var(
    const double & _arg)
  {
    this->ekf_y_acc_var = _arg;
    return *this;
  }
  Type & set__ekf_yaw_var(
    const double & _arg)
  {
    this->ekf_yaw_var = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::EKFErr_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::EKFErr_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::EKFErr_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::EKFErr_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::EKFErr_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::EKFErr_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::EKFErr_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::EKFErr_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::EKFErr_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::EKFErr_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__EKFErr
    std::shared_ptr<eufs_msgs::msg::EKFErr_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__EKFErr
    std::shared_ptr<eufs_msgs::msg::EKFErr_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const EKFErr_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->gps_x_vel_err != other.gps_x_vel_err) {
      return false;
    }
    if (this->gps_y_vel_err != other.gps_y_vel_err) {
      return false;
    }
    if (this->imu_x_acc_err != other.imu_x_acc_err) {
      return false;
    }
    if (this->imu_y_acc_err != other.imu_y_acc_err) {
      return false;
    }
    if (this->imu_yaw_err != other.imu_yaw_err) {
      return false;
    }
    if (this->ekf_x_vel_var != other.ekf_x_vel_var) {
      return false;
    }
    if (this->ekf_y_vel_var != other.ekf_y_vel_var) {
      return false;
    }
    if (this->ekf_x_acc_var != other.ekf_x_acc_var) {
      return false;
    }
    if (this->ekf_y_acc_var != other.ekf_y_acc_var) {
      return false;
    }
    if (this->ekf_yaw_var != other.ekf_yaw_var) {
      return false;
    }
    return true;
  }
  bool operator!=(const EKFErr_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct EKFErr_

// alias to use template instance with default allocator
using EKFErr =
  eufs_msgs::msg::EKFErr_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__EKF_ERR__STRUCT_HPP_
