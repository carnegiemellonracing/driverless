// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from eufs_msgs:srv/Register.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__SRV__DETAIL__REGISTER__STRUCT_HPP_
#define EUFS_MSGS__SRV__DETAIL__REGISTER__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__srv__Register_Request __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__srv__Register_Request __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Register_Request_
{
  using Type = Register_Request_<ContainerAllocator>;

  explicit Register_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->node_name = "";
      this->severity = 0;
    }
  }

  explicit Register_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : node_name(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->node_name = "";
      this->severity = 0;
    }
  }

  // field types and members
  using _node_name_type =
    std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other>;
  _node_name_type node_name;
  using _severity_type =
    uint8_t;
  _severity_type severity;

  // setters for named parameter idiom
  Type & set__node_name(
    const std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other> & _arg)
  {
    this->node_name = _arg;
    return *this;
  }
  Type & set__severity(
    const uint8_t & _arg)
  {
    this->severity = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::srv::Register_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::srv::Register_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::srv::Register_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::srv::Register_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::srv::Register_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::srv::Register_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::srv::Register_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::srv::Register_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::srv::Register_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::srv::Register_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__srv__Register_Request
    std::shared_ptr<eufs_msgs::srv::Register_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__srv__Register_Request
    std::shared_ptr<eufs_msgs::srv::Register_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Register_Request_ & other) const
  {
    if (this->node_name != other.node_name) {
      return false;
    }
    if (this->severity != other.severity) {
      return false;
    }
    return true;
  }
  bool operator!=(const Register_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Register_Request_

// alias to use template instance with default allocator
using Register_Request =
  eufs_msgs::srv::Register_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace eufs_msgs


#ifndef _WIN32
# define DEPRECATED__eufs_msgs__srv__Register_Response __attribute__((deprecated))
#else
# define DEPRECATED__eufs_msgs__srv__Register_Response __declspec(deprecated)
#endif

namespace eufs_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Register_Response_
{
  using Type = Register_Response_<ContainerAllocator>;

  explicit Register_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
    }
  }

  explicit Register_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->id = 0;
    }
  }

  // field types and members
  using _id_type =
    uint8_t;
  _id_type id;

  // setters for named parameter idiom
  Type & set__id(
    const uint8_t & _arg)
  {
    this->id = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    eufs_msgs::srv::Register_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const eufs_msgs::srv::Register_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<eufs_msgs::srv::Register_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<eufs_msgs::srv::Register_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::srv::Register_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::srv::Register_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      eufs_msgs::srv::Register_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<eufs_msgs::srv::Register_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<eufs_msgs::srv::Register_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<eufs_msgs::srv::Register_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__eufs_msgs__srv__Register_Response
    std::shared_ptr<eufs_msgs::srv::Register_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__eufs_msgs__srv__Register_Response
    std::shared_ptr<eufs_msgs::srv::Register_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Register_Response_ & other) const
  {
    if (this->id != other.id) {
      return false;
    }
    return true;
  }
  bool operator!=(const Register_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Register_Response_

// alias to use template instance with default allocator
using Register_Response =
  eufs_msgs::srv::Register_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace eufs_msgs

namespace eufs_msgs
{

namespace srv
{

struct Register
{
  using Request = eufs_msgs::srv::Register_Request;
  using Response = eufs_msgs::srv::Register_Response;
};

}  // namespace srv

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__SRV__DETAIL__REGISTER__STRUCT_HPP_
