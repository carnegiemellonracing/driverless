// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/BoundingBoxes.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__BOUNDING_BOXES__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__BOUNDING_BOXES__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'header'
// Member 'image_header'
#include "std_msgs/msg/detail/header__struct.hpp"
// Member 'bounding_boxes'
#include "eufs_msgs/msg/detail/bounding_box__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__BoundingBoxes __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__BoundingBoxes __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct BoundingBoxes_
{
  using Type = BoundingBoxes_<ContainerAllocator>;

  explicit BoundingBoxes_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    image_header(_init)
  {
    (void)_init;
  }

  explicit BoundingBoxes_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    image_header(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _image_header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _image_header_type image_header;
  using _bounding_boxes_type =
    std::vector<eufs_msgs::msg::BoundingBox_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::BoundingBox_<ContainerAllocator>>::other>;
  _bounding_boxes_type bounding_boxes;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__image_header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->image_header = _arg;
    return *this;
  }
  Type & set__bounding_boxes(
    const std::vector<eufs_msgs::msg::BoundingBox_<ContainerAllocator>, typename ContainerAllocator::template rebind<eufs_msgs::msg::BoundingBox_<ContainerAllocator>>::other> & _arg)
  {
    this->bounding_boxes = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::msg::BoundingBoxes_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::BoundingBoxes_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::BoundingBoxes_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::BoundingBoxes_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::BoundingBoxes_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::BoundingBoxes_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::BoundingBoxes_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::BoundingBoxes_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::BoundingBoxes_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::BoundingBoxes_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__BoundingBoxes
    std::shared_ptr<eufs_msgs::msg::BoundingBoxes_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__BoundingBoxes
    std::shared_ptr<eufs_msgs::msg::BoundingBoxes_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const BoundingBoxes_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->image_header != other.image_header) {
      return false;
    }
    if (this->bounding_boxes != other.bounding_boxes) {
      return false;
    }
    return true;
  }
  bool operator!=(const BoundingBoxes_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct BoundingBoxes_

// alias to use template instance with default allocator
using BoundingBoxes =
  eufs_msgs::msg::BoundingBoxes_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__BOUNDING_BOXES__STRUCT_HPP_
