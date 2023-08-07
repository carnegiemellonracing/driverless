// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/PathIntegralTiming.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/path_integral_timing__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/path_integral_timing__functions.h"
#include "eufs_msgs/msg/detail/path_integral_timing__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__PathIntegralTiming__init(message_memory);
}

void PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_fini_function(void * message_memory)
{
  eufs_msgs__msg__PathIntegralTiming__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_message_member_array[4] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PathIntegralTiming, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "average_time_between_poses",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PathIntegralTiming, average_time_between_poses),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "average_optimization_cycle_time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PathIntegralTiming, average_optimization_cycle_time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "average_sleep_time",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PathIntegralTiming, average_sleep_time),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_message_members = {
  "eufs_msgs__msg",  // message namespace
  "PathIntegralTiming",  // message name
  4,  // number of fields
  sizeof(eufs_msgs__msg__PathIntegralTiming),
  PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_message_member_array,  // message members
  PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_init_function,  // function to initialize message memory (memory has to be allocated)
  PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_message_type_support_handle = {
  0,
  &PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, PathIntegralTiming)() {
  PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_message_type_support_handle.typesupport_identifier) {
    PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &PathIntegralTiming__rosidl_typesupport_introspection_c__PathIntegralTiming_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
