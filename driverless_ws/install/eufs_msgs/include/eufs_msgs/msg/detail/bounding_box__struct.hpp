// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/BoundingBox.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__BOUNDING_BOX__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__BOUNDING_BOX__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__BoundingBox __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__BoundingBox __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct BoundingBox_
{
  using Type = BoundingBox_<ContainerAllocator>;

  explicit BoundingBox_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->color = "";
      this->probability = 0.0;
      this->type = 0l;
      this->xmin = 0.0;
      this->ymin = 0.0;
      this->xmax = 0.0;
      this->ymax = 0.0;
    }
  }

  explicit BoundingBox_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : color(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->color = "";
      this->probability = 0.0;
      this->type = 0l;
      this->xmin = 0.0;
      this->ymin = 0.0;
      this->xmax = 0.0;
      this->ymax = 0.0;
    }
  }

  // field types and members
  using _color_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _color_type color;
  using _probability_type =
    double;
  _probability_type probability;
  using _type_type =
    int32_t;
  _type_type type;
  using _xmin_type =
    double;
  _xmin_type xmin;
  using _ymin_type =
    double;
  _ymin_type ymin;
  using _xmax_type =
    double;
  _xmax_type xmax;
  using _ymax_type =
    double;
  _ymax_type ymax;

  // setters for named parameter idiom
  Type & set__color(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->color = _arg;
    return *this;
  }
  Type & set__probability(
    const double & _arg)
  {
    this->probability = _arg;
    return *this;
  }
  Type & set__type(
    const int32_t & _arg)
  {
    this->type = _arg;
    return *this;
  }
  Type & set__xmin(
    const double & _arg)
  {
    this->xmin = _arg;
    return *this;
  }
  Type & set__ymin(
    const double & _arg)
  {
    this->ymin = _arg;
    return *this;
  }
  Type & set__xmax(
    const double & _arg)
  {
    this->xmax = _arg;
    return *this;
  }
  Type & set__ymax(
    const double & _arg)
  {
    this->ymax = _arg;
    return *this;
  }

  // constant declarations
  static constexpr int32_t PIXEL =
    0;
  static constexpr int32_t PERCENTAGE =
    1;

  // pointer types
  using RawPtr =
    eufs_msgs::msg::BoundingBox_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::BoundingBox_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::BoundingBox_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::BoundingBox_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::BoundingBox_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::BoundingBox_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::BoundingBox_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::BoundingBox_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::BoundingBox_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::BoundingBox_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__BoundingBox
    std::shared_ptr<eufs_msgs::msg::BoundingBox_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__BoundingBox
    std::shared_ptr<eufs_msgs::msg::BoundingBox_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const BoundingBox_ & other) const
  {
    if (this->color != other.color) {
      return false;
    }
    if (this->probability != other.probability) {
      return false;
    }
    if (this->type != other.type) {
      return false;
    }
    if (this->xmin != other.xmin) {
      return false;
    }
    if (this->ymin != other.ymin) {
      return false;
    }
    if (this->xmax != other.xmax) {
      return false;
    }
    if (this->ymax != other.ymax) {
      return false;
    }
    return true;
  }
  bool operator!=(const BoundingBox_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct BoundingBox_

// alias to use template instance with default allocator
using BoundingBox =
  eufs_msgs::msg::BoundingBox_<std::allocator<void>>;

// constant definitions
template<typename ContainerAllocator>
constexpr int32_t BoundingBox_<ContainerAllocator>::PIXEL;
template<typename ContainerAllocator>
constexpr int32_t BoundingBox_<ContainerAllocator>::PERCENTAGE;

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__BOUNDING_BOX__STRUCT_HPP_
