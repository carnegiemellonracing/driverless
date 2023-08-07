// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:msg/TopicStatus.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__STRUCT_HPP_
#define EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__msg__TopicStatus __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__msg__TopicStatus __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct TopicStatus_
{
  using Type = TopicStatus_<ContainerAllocator>;

  explicit TopicStatus_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->topic = "";
      this->description = "";
      this->group = "";
      this->trigger_ebs = false;
      this->log_level = "";
      this->status = 0;
    }
  }

  explicit TopicStatus_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : topic(_alloc),
    description(_alloc),
    group(_alloc),
    log_level(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->topic = "";
      this->description = "";
      this->group = "";
      this->trigger_ebs = false;
      this->log_level = "";
      this->status = 0;
    }
  }

  // field types and members
  using _topic_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _topic_type topic;
  using _description_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _description_type description;
  using _group_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _group_type group;
  using _trigger_ebs_type =
    bool;
  _trigger_ebs_type trigger_ebs;
  using _log_level_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _log_level_type log_level;
  using _status_type =
    uint16_t;
  _status_type status;

  // setters for named parameter idiom
  Type & set__topic(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->topic = _arg;
    return *this;
  }
  Type & set__description(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->description = _arg;
    return *this;
  }
  Type & set__group(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->group = _arg;
    return *this;
  }
  Type & set__trigger_ebs(
    const bool & _arg)
  {
    this->trigger_ebs = _arg;
    return *this;
  }
  Type & set__log_level(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->log_level = _arg;
    return *this;
  }
  Type & set__status(
    const uint16_t & _arg)
  {
    this->status = _arg;
    return *this;
  }

  // constant declarations
  static constexpr uint16_t OFF =
    0u;
  static constexpr uint16_t PUBLISHING =
    1u;
  static constexpr uint16_t TIMEOUT_EXCEEDED =
    2u;

  // pointer types
  using RawPtr =
    eufs_msgs::msg::TopicStatus_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::msg::TopicStatus_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::msg::TopicStatus_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::msg::TopicStatus_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::TopicStatus_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::TopicStatus_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::msg::TopicStatus_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::msg::TopicStatus_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::msg::TopicStatus_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::msg::TopicStatus_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__msg__TopicStatus
    std::shared_ptr<eufs_msgs::msg::TopicStatus_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__msg__TopicStatus
    std::shared_ptr<eufs_msgs::msg::TopicStatus_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TopicStatus_ & other) const
  {
    if (this->topic != other.topic) {
      return false;
    }
    if (this->description != other.description) {
      return false;
    }
    if (this->group != other.group) {
      return false;
    }
    if (this->trigger_ebs != other.trigger_ebs) {
      return false;
    }
    if (this->log_level != other.log_level) {
      return false;
    }
    if (this->status != other.status) {
      return false;
    }
    return true;
  }
  bool operator!=(const TopicStatus_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TopicStatus_

// alias to use template instance with default allocator
using TopicStatus =
  eufs_msgs::msg::TopicStatus_<std::allocator<void>>;

// constant definitions
template<typename ContainerAllocator>
constexpr uint16_t TopicStatus_<ContainerAllocator>::OFF;
template<typename ContainerAllocator>
constexpr uint16_t TopicStatus_<ContainerAllocator>::PUBLISHING;
template<typename ContainerAllocator>
constexpr uint16_t TopicStatus_<ContainerAllocator>::TIMEOUT_EXCEEDED;

}  // namespace msg

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__STRUCT_HPP_
