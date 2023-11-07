// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from cmrdv_interfaces:msg/SimDataFrame.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__STRUCT_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'gt_cones'
#include "eufs_msgs/msg/detail/cone_array_with_covariance__struct.hpp"
// Member 'zed_left_img'
#include "sensor_msgs/msg/detail/image__struct.hpp"
// Member 'vlp16_pts'
// Member 'zed_pts'
#include "sensor_msgs/msg/detail/point_cloud2__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__cmrdv_interfaces__msg__SimDataFrame __attribute__((deprecated))
#else
# define DEPRECATED__cmrdv_interfaces__msg__SimDataFrame __declspec(deprecated)
#endif

namespace cmrdv_interfaces
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SimDataFrame_
{
  using Type = SimDataFrame_<ContainerAllocator>;

  explicit SimDataFrame_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : gt_cones(_init),
    zed_left_img(_init),
    vlp16_pts(_init),
    zed_pts(_init)
  {
    (void)_init;
  }

  explicit SimDataFrame_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : gt_cones(_alloc, _init),
    zed_left_img(_alloc, _init),
    vlp16_pts(_alloc, _init),
    zed_pts(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _gt_cones_type =
    eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator>;
  _gt_cones_type gt_cones;
  using _zed_left_img_type =
    sensor_msgs::msg::Image_<ContainerAllocator>;
  _zed_left_img_type zed_left_img;
  using _vlp16_pts_type =
    sensor_msgs::msg::PointCloud2_<ContainerAllocator>;
  _vlp16_pts_type vlp16_pts;
  using _zed_pts_type =
    sensor_msgs::msg::PointCloud2_<ContainerAllocator>;
  _zed_pts_type zed_pts;

  // setters for named parameter idiom
  Type & set__gt_cones(
    const eufs_msgs::msg::ConeArrayWithCovariance_<ContainerAllocator> & _arg)
  {
    this->gt_cones = _arg;
    return *this;
  }
  Type & set__zed_left_img(
    const sensor_msgs::msg::Image_<ContainerAllocator> & _arg)
  {
    this->zed_left_img = _arg;
    return *this;
  }
  Type & set__vlp16_pts(
    const sensor_msgs::msg::PointCloud2_<ContainerAllocator> & _arg)
  {
    this->vlp16_pts = _arg;
    return *this;
  }
  Type & set__zed_pts(
    const sensor_msgs::msg::PointCloud2_<ContainerAllocator> & _arg)
  {
    this->zed_pts = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator> *;
  using ConstRawPtr =
    const cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__cmrdv_interfaces__msg__SimDataFrame
    std::shared_ptr<cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__cmrdv_interfaces__msg__SimDataFrame
    std::shared_ptr<cmrdv_interfaces::msg::SimDataFrame_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SimDataFrame_ & other) const
  {
    if (this->gt_cones != other.gt_cones) {
      return false;
    }
    if (this->zed_left_img != other.zed_left_img) {
      return false;
    }
    if (this->vlp16_pts != other.vlp16_pts) {
      return false;
    }
    if (this->zed_pts != other.zed_pts) {
      return false;
    }
    return true;
  }
  bool operator!=(const SimDataFrame_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SimDataFrame_

// alias to use template instance with default allocator
using SimDataFrame =
  cmrdv_interfaces::msg::SimDataFrame_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__STRUCT_HPP_
