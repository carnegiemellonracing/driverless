// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cmrdv_interfaces:msg/DataFrame.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__STRUCT_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'zed_left_img'
#include "sensor_msgs/msg/detail/image__struct.hpp"
// Member 'zed_pts'
#include "sensor_msgs/msg/detail/point_cloud2__struct.hpp"
// Member 'sbg'
#include "sbg_driver/msg/detail/sbg_gps_pos__struct.hpp"
// Member 'imu'
#include "sbg_driver/msg/detail/sbg_imu_data__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__cmrdv_interfaces__msg__DataFrame __attribute__((deprecated))
#else
# define DEPRECATED__cmrdv_interfaces__msg__DataFrame __declspec(deprecated)
#endif

namespace cmrdv_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DataFrame_
{
  using Type = DataFrame_<ContainerAllocator>;

  explicit DataFrame_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : zed_left_img(_init),
    zed_pts(_init),
    sbg(_init),
    imu(_init)
  {
    (void)_init;
  }

  explicit DataFrame_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : zed_left_img(_alloc, _init),
    zed_pts(_alloc, _init),
    sbg(_alloc, _init),
    imu(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _zed_left_img_type =
    sensor_msgs::msg::Image_<ContainerAllocator>;
  _zed_left_img_type zed_left_img;
  using _zed_pts_type =
    sensor_msgs::msg::PointCloud2_<ContainerAllocator>;
  _zed_pts_type zed_pts;
  using _sbg_type =
    sbg_driver::msg::SbgGpsPos_<ContainerAllocator>;
  _sbg_type sbg;
  using _imu_type =
    sbg_driver::msg::SbgImuData_<ContainerAllocator>;
  _imu_type imu;

  // setters for named parameter idiom
  Type & set__zed_left_img(
    const sensor_msgs::msg::Image_<ContainerAllocator> & _arg)
  {
    this->zed_left_img = _arg;
    return *this;
  }
  Type & set__zed_pts(
    const sensor_msgs::msg::PointCloud2_<ContainerAllocator> & _arg)
  {
    this->zed_pts = _arg;
    return *this;
  }
  Type & set__sbg(
    const sbg_driver::msg::SbgGpsPos_<ContainerAllocator> & _arg)
  {
    this->sbg = _arg;
    return *this;
  }
  Type & set__imu(
    const sbg_driver::msg::SbgImuData_<ContainerAllocator> & _arg)
  {
    this->imu = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cmrdv_interfaces::msg::DataFrame_<ContainerAllocator> *;
  using ConstRawPtr =
    const cmrdv_interfaces::msg::DataFrame_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::DataFrame_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::DataFrame_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::DataFrame_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::DataFrame_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::DataFrame_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::DataFrame_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::DataFrame_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::DataFrame_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cmrdv_interfaces__msg__DataFrame
    std::shared_ptr<cmrdv_interfaces::msg::DataFrame_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cmrdv_interfaces__msg__DataFrame
    std::shared_ptr<cmrdv_interfaces::msg::DataFrame_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DataFrame_ & other) const
  {
    if (this->zed_left_img != other.zed_left_img) {
      return false;
    }
    if (this->zed_pts != other.zed_pts) {
      return false;
    }
    if (this->sbg != other.sbg) {
      return false;
    }
    if (this->imu != other.imu) {
      return false;
    }
    return true;
  }
  bool operator!=(const DataFrame_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DataFrame_

// alias to use template instance with default allocator
using DataFrame =
  cmrdv_interfaces::msg::DataFrame_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__STRUCT_HPP_
