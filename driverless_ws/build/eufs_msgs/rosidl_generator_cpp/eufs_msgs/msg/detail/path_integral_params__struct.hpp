// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/PathIntegralParams.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__PathIntegralParams __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__PathIntegralParams __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct PathIntegralParams_
{
  using Type = PathIntegralParams_<ContainerAllocator>;

  explicit PathIntegralParams_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->hz = 0ll;
      this->num_timesteps = 0ll;
      this->num_iters = 0ll;
      this->gamma = 0.0;
      this->init_steering = 0.0;
      this->init_throttle = 0.0;
      this->steering_var = 0.0;
      this->throttle_var = 0.0;
      this->max_throttle = 0.0;
      this->speed_coefficient = 0.0;
      this->track_coefficient = 0.0;
      this->max_slip_angle = 0.0;
      this->track_slop = 0.0;
      this->crash_coeff = 0.0;
      this->map_path = "";
      this->desired_speed = 0.0;
    }
  }

  explicit PathIntegralParams_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : map_path(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->hz = 0ll;
      this->num_timesteps = 0ll;
      this->num_iters = 0ll;
      this->gamma = 0.0;
      this->init_steering = 0.0;
      this->init_throttle = 0.0;
      this->steering_var = 0.0;
      this->throttle_var = 0.0;
      this->max_throttle = 0.0;
      this->speed_coefficient = 0.0;
      this->track_coefficient = 0.0;
      this->max_slip_angle = 0.0;
      this->track_slop = 0.0;
      this->crash_coeff = 0.0;
      this->map_path = "";
      this->desired_speed = 0.0;
    }
  }

  // field types and members
  using _hz_type =
    int64_t;
  _hz_type hz;
  using _num_timesteps_type =
    int64_t;
  _num_timesteps_type num_timesteps;
  using _num_iters_type =
    int64_t;
  _num_iters_type num_iters;
  using _gamma_type =
    double;
  _gamma_type gamma;
  using _init_steering_type =
    double;
  _init_steering_type init_steering;
  using _init_throttle_type =
    double;
  _init_throttle_type init_throttle;
  using _steering_var_type =
    double;
  _steering_var_type steering_var;
  using _throttle_var_type =
    double;
  _throttle_var_type throttle_var;
  using _max_throttle_type =
    double;
  _max_throttle_type max_throttle;
  using _speed_coefficient_type =
    double;
  _speed_coefficient_type speed_coefficient;
  using _track_coefficient_type =
    double;
  _track_coefficient_type track_coefficient;
  using _max_slip_angle_type =
    double;
  _max_slip_angle_type max_slip_angle;
  using _track_slop_type =
    double;
  _track_slop_type track_slop;
  using _crash_coeff_type =
    double;
  _crash_coeff_type crash_coeff;
  using _map_path_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _map_path_type map_path;
  using _desired_speed_type =
    double;
  _desired_speed_type desired_speed;

  // setters for named parameter idiom
  Type & set__hz(
    const int64_t & _arg)
  {
    this->hz = _arg;
    return *this;
  }
  Type & set__num_timesteps(
    const int64_t & _arg)
  {
    this->num_timesteps = _arg;
    return *this;
  }
  Type & set__num_iters(
    const int64_t & _arg)
  {
    this->num_iters = _arg;
    return *this;
  }
  Type & set__gamma(
    const double & _arg)
  {
    this->gamma = _arg;
    return *this;
  }
  Type & set__init_steering(
    const double & _arg)
  {
    this->init_steering = _arg;
    return *this;
  }
  Type & set__init_throttle(
    const double & _arg)
  {
    this->init_throttle = _arg;
    return *this;
  }
  Type & set__steering_var(
    const double & _arg)
  {
    this->steering_var = _arg;
    return *this;
  }
  Type & set__throttle_var(
    const double & _arg)
  {
    this->throttle_var = _arg;
    return *this;
  }
  Type & set__max_throttle(
    const double & _arg)
  {
    this->max_throttle = _arg;
    return *this;
  }
  Type & set__speed_coefficient(
    const double & _arg)
  {
    this->speed_coefficient = _arg;
    return *this;
  }
  Type & set__track_coefficient(
    const double & _arg)
  {
    this->track_coefficient = _arg;
    return *this;
  }
  Type & set__max_slip_angle(
    const double & _arg)
  {
    this->max_slip_angle = _arg;
    return *this;
  }
  Type & set__track_slop(
    const double & _arg)
  {
    this->track_slop = _arg;
    return *this;
  }
  Type & set__crash_coeff(
    const double & _arg)
  {
    this->crash_coeff = _arg;
    return *this;
  }
  Type & set__map_path(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->map_path = _arg;
    return *this;
  }
  Type & set__desired_speed(
    const double & _arg)
  {
    this->desired_speed = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::PathIntegralParams_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::PathIntegralParams_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::PathIntegralParams_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::PathIntegralParams_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PathIntegralParams_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PathIntegralParams_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::PathIntegralParams_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::PathIntegralParams_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::PathIntegralParams_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::PathIntegralParams_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__PathIntegralParams
    std::shared_ptr<eufs_msgs::msg::PathIntegralParams_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__PathIntegralParams
    std::shared_ptr<eufs_msgs::msg::PathIntegralParams_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const PathIntegralParams_ & other) const
  {
    if (this->hz != other.hz) {
      return false;
    }
    if (this->num_timesteps != other.num_timesteps) {
      return false;
    }
    if (this->num_iters != other.num_iters) {
      return false;
    }
    if (this->gamma != other.gamma) {
      return false;
    }
    if (this->init_steering != other.init_steering) {
      return false;
    }
    if (this->init_throttle != other.init_throttle) {
      return false;
    }
    if (this->steering_var != other.steering_var) {
      return false;
    }
    if (this->throttle_var != other.throttle_var) {
      return false;
    }
    if (this->max_throttle != other.max_throttle) {
      return false;
    }
    if (this->speed_coefficient != other.speed_coefficient) {
      return false;
    }
    if (this->track_coefficient != other.track_coefficient) {
      return false;
    }
    if (this->max_slip_angle != other.max_slip_angle) {
      return false;
    }
    if (this->track_slop != other.track_slop) {
      return false;
    }
    if (this->crash_coeff != other.crash_coeff) {
      return false;
    }
    if (this->map_path != other.map_path) {
      return false;
    }
    if (this->desired_speed != other.desired_speed) {
      return false;
    }
    return true;
  }
  bool operator!=(const PathIntegralParams_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct PathIntegralParams_

// alias to use template instance with default allocator
using PathIntegralParams =
  eufs_msgs::msg::PathIntegralParams_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__STRUCT_HPP_
