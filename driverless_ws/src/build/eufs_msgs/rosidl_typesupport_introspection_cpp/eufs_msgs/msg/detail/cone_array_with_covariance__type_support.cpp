// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/ConeArrayWithCovariance.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "eufs_msgs/msg/detail/cone_array_with_covariance__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace eufs_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void ConeArrayWithCovariance_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) eufs_msgs::msg::ConeArrayWithCovariance(_init);
}

void ConeArrayWithCovariance_fini_function(void * message_memory)
{
  auto typed_message = static_cast<eufs_msgs::msg::ConeArrayWithCovariance *>(message_memory);
  typed_message->~ConeArrayWithCovariance();
}

size_t size_function__ConeArrayWithCovariance__blue_cones(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ConeArrayWithCovariance__blue_cones(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return &member[index];
}

void * get_function__ConeArrayWithCovariance__blue_cones(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return &member[index];
}

void resize_function__ConeArrayWithCovariance__blue_cones(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ConeArrayWithCovariance__yellow_cones(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ConeArrayWithCovariance__yellow_cones(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return &member[index];
}

void * get_function__ConeArrayWithCovariance__yellow_cones(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return &member[index];
}

void resize_function__ConeArrayWithCovariance__yellow_cones(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ConeArrayWithCovariance__orange_cones(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ConeArrayWithCovariance__orange_cones(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return &member[index];
}

void * get_function__ConeArrayWithCovariance__orange_cones(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return &member[index];
}

void resize_function__ConeArrayWithCovariance__orange_cones(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ConeArrayWithCovariance__big_orange_cones(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ConeArrayWithCovariance__big_orange_cones(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return &member[index];
}

void * get_function__ConeArrayWithCovariance__big_orange_cones(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return &member[index];
}

void resize_function__ConeArrayWithCovariance__big_orange_cones(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ConeArrayWithCovariance__unknown_color_cones(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ConeArrayWithCovariance__unknown_color_cones(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return &member[index];
}

void * get_function__ConeArrayWithCovariance__unknown_color_cones(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  return &member[index];
}

void resize_function__ConeArrayWithCovariance__unknown_color_cones(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<eufs_msgs::msg::ConeWithCovariance> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ConeArrayWithCovariance_message_member_array[6] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArrayWithCovariance, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "blue_cones",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<eufs_msgs::msg::ConeWithCovariance>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArrayWithCovariance, blue_cones),  // bytes offset in struct
    nullptr,  // default value
    size_function__ConeArrayWithCovariance__blue_cones,  // size() function pointer
    get_const_function__ConeArrayWithCovariance__blue_cones,  // get_const(index) function pointer
    get_function__ConeArrayWithCovariance__blue_cones,  // get(index) function pointer
    resize_function__ConeArrayWithCovariance__blue_cones  // resize(index) function pointer
  },
  {
    "yellow_cones",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<eufs_msgs::msg::ConeWithCovariance>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArrayWithCovariance, yellow_cones),  // bytes offset in struct
    nullptr,  // default value
    size_function__ConeArrayWithCovariance__yellow_cones,  // size() function pointer
    get_const_function__ConeArrayWithCovariance__yellow_cones,  // get_const(index) function pointer
    get_function__ConeArrayWithCovariance__yellow_cones,  // get(index) function pointer
    resize_function__ConeArrayWithCovariance__yellow_cones  // resize(index) function pointer
  },
  {
    "orange_cones",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<eufs_msgs::msg::ConeWithCovariance>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArrayWithCovariance, orange_cones),  // bytes offset in struct
    nullptr,  // default value
    size_function__ConeArrayWithCovariance__orange_cones,  // size() function pointer
    get_const_function__ConeArrayWithCovariance__orange_cones,  // get_const(index) function pointer
    get_function__ConeArrayWithCovariance__orange_cones,  // get(index) function pointer
    resize_function__ConeArrayWithCovariance__orange_cones  // resize(index) function pointer
  },
  {
    "big_orange_cones",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<eufs_msgs::msg::ConeWithCovariance>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArrayWithCovariance, big_orange_cones),  // bytes offset in struct
    nullptr,  // default value
    size_function__ConeArrayWithCovariance__big_orange_cones,  // size() function pointer
    get_const_function__ConeArrayWithCovariance__big_orange_cones,  // get_const(index) function pointer
    get_function__ConeArrayWithCovariance__big_orange_cones,  // get(index) function pointer
    resize_function__ConeArrayWithCovariance__big_orange_cones  // resize(index) function pointer
  },
  {
    "unknown_color_cones",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<eufs_msgs::msg::ConeWithCovariance>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArrayWithCovariance, unknown_color_cones),  // bytes offset in struct
    nullptr,  // default value
    size_function__ConeArrayWithCovariance__unknown_color_cones,  // size() function pointer
    get_const_function__ConeArrayWithCovariance__unknown_color_cones,  // get_const(index) function pointer
    get_function__ConeArrayWithCovariance__unknown_color_cones,  // get(index) function pointer
    resize_function__ConeArrayWithCovariance__unknown_color_cones  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ConeArrayWithCovariance_message_members = {
  "eufs_msgs::msg",  // message namespace
  "ConeArrayWithCovariance",  // message name
  6,  // number of fields
  sizeof(eufs_msgs::msg::ConeArrayWithCovariance),
  ConeArrayWithCovariance_message_member_array,  // message members
  ConeArrayWithCovariance_init_function,  // function to initialize message memory (memory has to be allocated)
  ConeArrayWithCovariance_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ConeArrayWithCovariance_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ConeArrayWithCovariance_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace eufs_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<eufs_msgs::msg::ConeArrayWithCovariance>()
{
  return &::eufs_msgs::msg::rosidl_typesupport_introspection_cpp::ConeArrayWithCovariance_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, eufs_msgs, msg, ConeArrayWithCovariance)() {
  return &::eufs_msgs::msg::rosidl_typesupport_introspection_cpp::ConeArrayWithCovariance_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
