// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/ConeArray.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "eufs_msgs/msg/detail/cone_array__struct.hpp"
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

void ConeArray_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) eufs_msgs::msg::ConeArray(_init);
}

void ConeArray_fini_function(void * message_memory)
{
  auto typed_message = static_cast<eufs_msgs::msg::ConeArray *>(message_memory);
  typed_message->~ConeArray();
}

size_t size_function__ConeArray__blue_cones(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ConeArray__blue_cones(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void * get_function__ConeArray__blue_cones(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void resize_function__ConeArray__blue_cones(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ConeArray__yellow_cones(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ConeArray__yellow_cones(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void * get_function__ConeArray__yellow_cones(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void resize_function__ConeArray__yellow_cones(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ConeArray__orange_cones(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ConeArray__orange_cones(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void * get_function__ConeArray__orange_cones(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void resize_function__ConeArray__orange_cones(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ConeArray__big_orange_cones(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ConeArray__big_orange_cones(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void * get_function__ConeArray__big_orange_cones(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void resize_function__ConeArray__big_orange_cones(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  member->resize(size);
}

size_t size_function__ConeArray__unknown_color_cones(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ConeArray__unknown_color_cones(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void * get_function__ConeArray__unknown_color_cones(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  return &member[index];
}

void resize_function__ConeArray__unknown_color_cones(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<geometry_msgs::msg::Point> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ConeArray_message_member_array[6] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArray, header),  // bytes offset in struct
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
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::Point>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArray, blue_cones),  // bytes offset in struct
    nullptr,  // default value
    size_function__ConeArray__blue_cones,  // size() function pointer
    get_const_function__ConeArray__blue_cones,  // get_const(index) function pointer
    get_function__ConeArray__blue_cones,  // get(index) function pointer
    resize_function__ConeArray__blue_cones  // resize(index) function pointer
  },
  {
    "yellow_cones",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::Point>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArray, yellow_cones),  // bytes offset in struct
    nullptr,  // default value
    size_function__ConeArray__yellow_cones,  // size() function pointer
    get_const_function__ConeArray__yellow_cones,  // get_const(index) function pointer
    get_function__ConeArray__yellow_cones,  // get(index) function pointer
    resize_function__ConeArray__yellow_cones  // resize(index) function pointer
  },
  {
    "orange_cones",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::Point>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArray, orange_cones),  // bytes offset in struct
    nullptr,  // default value
    size_function__ConeArray__orange_cones,  // size() function pointer
    get_const_function__ConeArray__orange_cones,  // get_const(index) function pointer
    get_function__ConeArray__orange_cones,  // get(index) function pointer
    resize_function__ConeArray__orange_cones  // resize(index) function pointer
  },
  {
    "big_orange_cones",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::Point>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArray, big_orange_cones),  // bytes offset in struct
    nullptr,  // default value
    size_function__ConeArray__big_orange_cones,  // size() function pointer
    get_const_function__ConeArray__big_orange_cones,  // get_const(index) function pointer
    get_function__ConeArray__big_orange_cones,  // get(index) function pointer
    resize_function__ConeArray__big_orange_cones  // resize(index) function pointer
  },
  {
    "unknown_color_cones",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<geometry_msgs::msg::Point>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::ConeArray, unknown_color_cones),  // bytes offset in struct
    nullptr,  // default value
    size_function__ConeArray__unknown_color_cones,  // size() function pointer
    get_const_function__ConeArray__unknown_color_cones,  // get_const(index) function pointer
    get_function__ConeArray__unknown_color_cones,  // get(index) function pointer
    resize_function__ConeArray__unknown_color_cones  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ConeArray_message_members = {
  "eufs_msgs::msg",  // message namespace
  "ConeArray",  // message name
  6,  // number of fields
  sizeof(eufs_msgs::msg::ConeArray),
  ConeArray_message_member_array,  // message members
  ConeArray_init_function,  // function to initialize message memory (memory has to be allocated)
  ConeArray_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ConeArray_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ConeArray_message_members,
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
get_message_type_support_handle<eufs_msgs::msg::ConeArray>()
{
  return &::eufs_msgs::msg::rosidl_typesupport_introspection_cpp::ConeArray_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, eufs_msgs, msg, ConeArray)() {
  return &::eufs_msgs::msg::rosidl_typesupport_introspection_cpp::ConeArray_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
