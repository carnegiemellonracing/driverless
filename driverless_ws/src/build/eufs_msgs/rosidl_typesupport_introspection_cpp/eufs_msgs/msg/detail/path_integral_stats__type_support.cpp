// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/PathIntegralStats.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "eufs_msgs/msg/detail/path_integral_stats__struct.hpp"
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

void PathIntegralStats_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) eufs_msgs::msg::PathIntegralStats(_init);
}

void PathIntegralStats_fini_function(void * message_memory)
{
  auto typed_message = static_cast<eufs_msgs::msg::PathIntegralStats *>(message_memory);
  typed_message->~PathIntegralStats();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember PathIntegralStats_message_member_array[4] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::PathIntegralStats, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "tag",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::PathIntegralStats, tag),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "params",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<eufs_msgs::msg::PathIntegralParams>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::PathIntegralStats, params),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "stats",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<eufs_msgs::msg::LapStats>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs::msg::PathIntegralStats, stats),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers PathIntegralStats_message_members = {
  "eufs_msgs::msg",  // message namespace
  "PathIntegralStats",  // message name
  4,  // number of fields
  sizeof(eufs_msgs::msg::PathIntegralStats),
  PathIntegralStats_message_member_array,  // message members
  PathIntegralStats_init_function,  // function to initialize message memory (memory has to be allocated)
  PathIntegralStats_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t PathIntegralStats_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &PathIntegralStats_message_members,
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
get_message_type_support_handle<eufs_msgs::msg::PathIntegralStats>()
{
  return &::eufs_msgs::msg::rosidl_typesupport_introspection_cpp::PathIntegralStats_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, eufs_msgs, msg, PathIntegralStats)() {
  return &::eufs_msgs::msg::rosidl_typesupport_introspection_cpp::PathIntegralStats_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
