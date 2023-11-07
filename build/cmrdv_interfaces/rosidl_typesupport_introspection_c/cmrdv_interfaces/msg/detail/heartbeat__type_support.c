// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cmrdv_interfaces:msg/Heartbeat.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cmrdv_interfaces/msg/detail/heartbeat__rosidl_typesupport_introspection_c.h"
#include "cmrdv_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cmrdv_interfaces/msg/detail/heartbeat__functions.h"
#include "cmrdv_interfaces/msg/detail/heartbeat__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cmrdv_interfaces__msg__Heartbeat__init(message_memory);
}

void Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_fini_function(void * message_memory)
{
  cmrdv_interfaces__msg__Heartbeat__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__Heartbeat, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "status",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__Heartbeat, status),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_message_members = {
  "cmrdv_interfaces__msg",  // message namespace
  "Heartbeat",  // message name
  2,  // number of fields
  sizeof(cmrdv_interfaces__msg__Heartbeat),
  Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_message_member_array,  // message members
  Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_init_function,  // function to initialize message memory (memory has to be allocated)
  Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_message_type_support_handle = {
  0,
  &Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cmrdv_interfaces, msg, Heartbeat)() {
  Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_message_type_support_handle.typesupport_identifier) {
    Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &Heartbeat__rosidl_typesupport_introspection_c__Heartbeat_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
