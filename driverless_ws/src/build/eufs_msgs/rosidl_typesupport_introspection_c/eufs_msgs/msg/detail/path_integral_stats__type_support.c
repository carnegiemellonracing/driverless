// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/PathIntegralStats.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/path_integral_stats__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/path_integral_stats__functions.h"
#include "eufs_msgs/msg/detail/path_integral_stats__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `tag`
#include "rosidl_runtime_c/string_functions.h"
// Member `params`
#include "eufs_msgs/msg/path_integral_params.h"
// Member `params`
#include "eufs_msgs/msg/detail/path_integral_params__rosidl_typesupport_introspection_c.h"
// Member `stats`
#include "eufs_msgs/msg/lap_stats.h"
// Member `stats`
#include "eufs_msgs/msg/detail/lap_stats__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__PathIntegralStats__init(message_memory);
}

void PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_fini_function(void * message_memory)
{
  eufs_msgs__msg__PathIntegralStats__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_message_member_array[4] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PathIntegralStats, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "tag",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PathIntegralStats, tag),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "params",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PathIntegralStats, params),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "stats",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PathIntegralStats, stats),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_message_members = {
  "eufs_msgs__msg",  // message namespace
  "PathIntegralStats",  // message name
  4,  // number of fields
  sizeof(eufs_msgs__msg__PathIntegralStats),
  PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_message_member_array,  // message members
  PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_init_function,  // function to initialize message memory (memory has to be allocated)
  PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_message_type_support_handle = {
  0,
  &PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, PathIntegralStats)() {
  PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, PathIntegralParams)();
  PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, LapStats)();
  if (!PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_message_type_support_handle.typesupport_identifier) {
    PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &PathIntegralStats__rosidl_typesupport_introspection_c__PathIntegralStats_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
