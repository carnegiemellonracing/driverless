// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/WheelSpeeds.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/wheel_speeds__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/wheel_speeds__functions.h"
#include "eufs_msgs/msg/detail/wheel_speeds__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__WheelSpeeds__init(message_memory);
}

void WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_fini_function(void * message_memory)
{
  eufs_msgs__msg__WheelSpeeds__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_message_member_array[5] = {
  {
    "steering",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WheelSpeeds, steering),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "lf_speed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WheelSpeeds, lf_speed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "rf_speed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WheelSpeeds, rf_speed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "lb_speed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WheelSpeeds, lb_speed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "rb_speed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WheelSpeeds, rb_speed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_message_members = {
  "eufs_msgs__msg",  // message namespace
  "WheelSpeeds",  // message name
  5,  // number of fields
  sizeof(eufs_msgs__msg__WheelSpeeds),
  WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_message_member_array,  // message members
  WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_init_function,  // function to initialize message memory (memory has to be allocated)
  WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_message_type_support_handle = {
  0,
  &WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, WheelSpeeds)() {
  if (!WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_message_type_support_handle.typesupport_identifier) {
    WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &WheelSpeeds__rosidl_typesupport_introspection_c__WheelSpeeds_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
