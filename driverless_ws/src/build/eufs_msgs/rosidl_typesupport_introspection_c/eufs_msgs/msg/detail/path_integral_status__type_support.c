// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/PathIntegralStatus.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/path_integral_status__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/path_integral_status__functions.h"
#include "eufs_msgs/msg/detail/path_integral_status__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `info`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__PathIntegralStatus__init(message_memory);
}

void PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_fini_function(void * message_memory)
{
  eufs_msgs__msg__PathIntegralStatus__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_message_member_array[3] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PathIntegralStatus, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "info",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PathIntegralStatus, info),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "status",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PathIntegralStatus, status),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_message_members = {
  "eufs_msgs__msg",  // message namespace
  "PathIntegralStatus",  // message name
  3,  // number of fields
  sizeof(eufs_msgs__msg__PathIntegralStatus),
  PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_message_member_array,  // message members
  PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_init_function,  // function to initialize message memory (memory has to be allocated)
  PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_message_type_support_handle = {
  0,
  &PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, PathIntegralStatus)() {
  PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_message_type_support_handle.typesupport_identifier) {
    PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &PathIntegralStatus__rosidl_typesupport_introspection_c__PathIntegralStatus_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
