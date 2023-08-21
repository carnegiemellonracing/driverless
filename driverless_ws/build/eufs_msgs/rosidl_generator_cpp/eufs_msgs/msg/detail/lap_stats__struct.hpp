// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/LapStats.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__LAP_STATS__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__LAP_STATS__STRUCT_HPP_

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
# define DEPRECATED__eufs_msgs__msg__LapStats __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__LapStats __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct LapStats_
{
  using Type = LapStats_<ContainerAllocator>;

  explicit LapStats_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->lap_number = 0ll;
      this->lap_time = 0.0;
      this->avg_speed = 0.0;
      this->max_speed = 0.0;
      this->speed_var = 0.0;
      this->max_slip = 0.0;
      this->max_lateral_accel = 0.0;
      this->normalized_deviation_mse = 0.0;
      this->deviation_var = 0.0;
      this->max_deviation = 0.0;
    }
  }

  explicit LapStats_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->lap_number = 0ll;
      this->lap_time = 0.0;
      this->avg_speed = 0.0;
      this->max_speed = 0.0;
      this->speed_var = 0.0;
      this->max_slip = 0.0;
      this->max_lateral_accel = 0.0;
      this->normalized_deviation_mse = 0.0;
      this->deviation_var = 0.0;
      this->max_deviation = 0.0;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _lap_number_type =
    int64_t;
  _lap_number_type lap_number;
  using _lap_time_type =
    double;
  _lap_time_type lap_time;
  using _avg_speed_type =
    double;
  _avg_speed_type avg_speed;
  using _max_speed_type =
    double;
  _max_speed_type max_speed;
  using _speed_var_type =
    double;
  _speed_var_type speed_var;
  using _max_slip_type =
    double;
  _max_slip_type max_slip;
  using _max_lateral_accel_type =
    double;
  _max_lateral_accel_type max_lateral_accel;
  using _normalized_deviation_mse_type =
    double;
  _normalized_deviation_mse_type normalized_deviation_mse;
  using _deviation_var_type =
    double;
  _deviation_var_type deviation_var;
  using _max_deviation_type =
    double;
  _max_deviation_type max_deviation;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__lap_number(
    const int64_t & _arg)
  {
    this->lap_number = _arg;
    return *this;
  }
  Type & set__lap_time(
    const double & _arg)
  {
    this->lap_time = _arg;
    return *this;
  }
  Type & set__avg_speed(
    const double & _arg)
  {
    this->avg_speed = _arg;
    return *this;
  }
  Type & set__max_speed(
    const double & _arg)
  {
    this->max_speed = _arg;
    return *this;
  }
  Type & set__speed_var(
    const double & _arg)
  {
    this->speed_var = _arg;
    return *this;
  }
  Type & set__max_slip(
    const double & _arg)
  {
    this->max_slip = _arg;
    return *this;
  }
  Type & set__max_lateral_accel(
    const double & _arg)
  {
    this->max_lateral_accel = _arg;
    return *this;
  }
  Type & set__normalized_deviation_mse(
    const double & _arg)
  {
    this->normalized_deviation_mse = _arg;
    return *this;
  }
  Type & set__deviation_var(
    const double & _arg)
  {
    this->deviation_var = _arg;
    return *this;
  }
  Type & set__max_deviation(
    const double & _arg)
  {
    this->max_deviation = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::LapStats_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::LapStats_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::LapStats_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::LapStats_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::LapStats_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::LapStats_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::LapStats_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::LapStats_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::LapStats_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::LapStats_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__LapStats
    std::shared_ptr<eufs_msgs::msg::LapStats_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__LapStats
    std::shared_ptr<eufs_msgs::msg::LapStats_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const LapStats_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->lap_number != other.lap_number) {
      return false;
    }
    if (this->lap_time != other.lap_time) {
      return false;
    }
    if (this->avg_speed != other.avg_speed) {
      return false;
    }
    if (this->max_speed != other.max_speed) {
      return false;
    }
    if (this->speed_var != other.speed_var) {
      return false;
    }
    if (this->max_slip != other.max_slip) {
      return false;
    }
    if (this->max_lateral_accel != other.max_lateral_accel) {
      return false;
    }
    if (this->normalized_deviation_mse != other.normalized_deviation_mse) {
      return false;
    }
    if (this->deviation_var != other.deviation_var) {
      return false;
    }
    if (this->max_deviation != other.max_deviation) {
      return false;
    }
    return true;
  }
  bool operator!=(const LapStats_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct LapStats_

// alias to use template instance with default allocator
using LapStats =
  eufs_msgs::msg::LapStats_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__LAP_STATS__STRUCT_HPP_
