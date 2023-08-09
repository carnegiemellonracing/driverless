// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/Costmap.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__COSTMAP__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__COSTMAP__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__Costmap __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__Costmap __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct Costmap_
{
  using Type = Costmap_<ContainerAllocator>;

  explicit Costmap_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x_bounds_min = 0.0;
      this->x_bounds_max = 0.0;
      this->y_bounds_min = 0.0;
      this->y_bounds_max = 0.0;
      this->pixels_per_meter = 0.0;
    }
  }

  explicit Costmap_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x_bounds_min = 0.0;
      this->x_bounds_max = 0.0;
      this->y_bounds_min = 0.0;
      this->y_bounds_max = 0.0;
      this->pixels_per_meter = 0.0;
    }
  }

  // field types and members
  using _x_bounds_min_type =
    double;
  _x_bounds_min_type x_bounds_min;
  using _x_bounds_max_type =
    double;
  _x_bounds_max_type x_bounds_max;
  using _y_bounds_min_type =
    double;
  _y_bounds_min_type y_bounds_min;
  using _y_bounds_max_type =
    double;
  _y_bounds_max_type y_bounds_max;
  using _pixels_per_meter_type =
    double;
  _pixels_per_meter_type pixels_per_meter;
  using _channel0_type =
    std::vector<float, typename ContainerAllocator::template rebind<float>::other>;
  _channel0_type channel0;
  using _channel1_type =
    std::vector<float, typename ContainerAllocator::template rebind<float>::other>;
  _channel1_type channel1;
  using _channel2_type =
    std::vector<float, typename ContainerAllocator::template rebind<float>::other>;
  _channel2_type channel2;
  using _channel3_type =
    std::vector<float, typename ContainerAllocator::template rebind<float>::other>;
  _channel3_type channel3;

  // setters for named parameter idiom
  Type & set__x_bounds_min(
    const double & _arg)
  {
    this->x_bounds_min = _arg;
    return *this;
  }
  Type & set__x_bounds_max(
    const double & _arg)
  {
    this->x_bounds_max = _arg;
    return *this;
  }
  Type & set__y_bounds_min(
    const double & _arg)
  {
    this->y_bounds_min = _arg;
    return *this;
  }
  Type & set__y_bounds_max(
    const double & _arg)
  {
    this->y_bounds_max = _arg;
    return *this;
  }
  Type & set__pixels_per_meter(
    const double & _arg)
  {
    this->pixels_per_meter = _arg;
    return *this;
  }
  Type & set__channel0(
    const std::vector<float, typename ContainerAllocator::template rebind<float>::other> & _arg)
  {
    this->channel0 = _arg;
    return *this;
  }
  Type & set__channel1(
    const std::vector<float, typename ContainerAllocator::template rebind<float>::other> & _arg)
  {
    this->channel1 = _arg;
    return *this;
  }
  Type & set__channel2(
    const std::vector<float, typename ContainerAllocator::template rebind<float>::other> & _arg)
  {
    this->channel2 = _arg;
    return *this;
  }
  Type & set__channel3(
    const std::vector<float, typename ContainerAllocator::template rebind<float>::other> & _arg)
  {
    this->channel3 = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::Costmap_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::Costmap_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::Costmap_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::Costmap_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::Costmap_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::Costmap_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::Costmap_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::Costmap_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::Costmap_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::Costmap_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__Costmap
    std::shared_ptr<eufs_msgs::msg::Costmap_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__Costmap
    std::shared_ptr<eufs_msgs::msg::Costmap_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Costmap_ & other) const
  {
    if (this->x_bounds_min != other.x_bounds_min) {
      return false;
    }
    if (this->x_bounds_max != other.x_bounds_max) {
      return false;
    }
    if (this->y_bounds_min != other.y_bounds_min) {
      return false;
    }
    if (this->y_bounds_max != other.y_bounds_max) {
      return false;
    }
    if (this->pixels_per_meter != other.pixels_per_meter) {
      return false;
    }
    if (this->channel0 != other.channel0) {
      return false;
    }
    if (this->channel1 != other.channel1) {
      return false;
    }
    if (this->channel2 != other.channel2) {
      return false;
    }
    if (this->channel3 != other.channel3) {
      return false;
    }
    return true;
  }
  bool operator!=(const Costmap_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Costmap_

// alias to use template instance with default allocator
using Costmap =
  eufs_msgs::msg::Costmap_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__COSTMAP__STRUCT_HPP_
