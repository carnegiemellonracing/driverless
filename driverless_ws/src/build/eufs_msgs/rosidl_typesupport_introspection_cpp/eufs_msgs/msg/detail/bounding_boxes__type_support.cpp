// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/BoundingBoxes.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "eufs_msgs/msg/detail/bounding_boxes__struct.hpp"
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

void BoundingBoxes_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) eufs_msgs::msg::BoundingBoxes(_init);
}

void BoundingBoxes_fini_function(void * message_memory)
{
  auto typed_message = static_cast<eufs_msgs::msg::BoundingBoxes *>(message_memory);
  typed_message->~BoundingBoxes();
}

size_t size_function__BoundingBoxes__bounding_boxes(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<eufs_msgs::msg::BoundingBox> *>(untyped_member);
  return member->size();
}

const void * get_const_function__BoundingBoxes__bounding_boxes(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<eufs_msgs::msg::BoundingBox> *>(untyped_member);
  return &member[index];
}

void * get_function__BoundingBoxes__bounding_boxes(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<eufs_msgs::msg::BoundingBox> *>(untyped_member);
  return &member[index];
}

void resize_function__BoundingBoxes__bounding_boxes(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<eufs_msgs::msg::BoundingBox> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember BoundingBoxes_message_member_array[3] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::BoundingBoxes, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "image_header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::BoundingBoxes, image_header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "bounding_boxes",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<eufs_msgs::msg::BoundingBox>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::BoundingBoxes, bounding_boxes),  // bytes offset in struct
    nullptr,  // default value
    size_function__BoundingBoxes__bounding_boxes,  // size() function pointer
    get_const_function__BoundingBoxes__bounding_boxes,  // get_const(index) function pointer
    get_function__BoundingBoxes__bounding_boxes,  // get(index) function pointer
    resize_function__BoundingBoxes__bounding_boxes  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers BoundingBoxes_message_members = {
  "eufs_msgs::msg",  // message namespace
  "BoundingBoxes",  // message name
  3,  // number of fields
  sizeof(eufs_msgs::msg::BoundingBoxes),
  BoundingBoxes_message_member_array,  // message members
  BoundingBoxes_init_function,  // function to initialize message memory (memory has to be allocated)
  BoundingBoxes_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t BoundingBoxes_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &BoundingBoxes_message_members,
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
get_message_type_support_handle<eufs_msgs::msg::BoundingBoxes>()
{
  return &::eufs_msgs::msg::rosidl_typesupport_introspection_cpp::BoundingBoxes_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, eufs_msgs, msg, BoundingBoxes)() {
  return &::eufs_msgs::msg::rosidl_typesupport_introspection_cpp::BoundingBoxes_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
