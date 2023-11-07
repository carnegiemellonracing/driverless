// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cmrdv_interfaces:msg/ControlAction.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cmrdv_interfaces/msg/detail/control_action__rosidl_typesupport_introspection_c.h"
#include "cmrdv_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cmrdv_interfaces/msg/detail/control_action__functions.h"
#include "cmrdv_interfaces/msg/detail/control_action__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void ControlAction__rosidl_typesupport_introspection_c__ControlAction_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cmrdv_interfaces__msg__ControlAction__init(message_memory);
}

void ControlAction__rosidl_typesupport_introspection_c__ControlAction_fini_function(void * message_memory)
{
  cmrdv_interfaces__msg__ControlAction__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember ControlAction__rosidl_typesupport_introspection_c__ControlAction_message_member_array[2] = {
  {
    "wheel_speed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__ControlAction, wheel_speed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "swangle",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__ControlAction, swangle),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ControlAction__rosidl_typesupport_introspection_c__ControlAction_message_members = {
  "cmrdv_interfaces__msg",  // message namespace
  "ControlAction",  // message name
  2,  // number of fields
  sizeof(cmrdv_interfaces__msg__ControlAction),
  ControlAction__rosidl_typesupport_introspection_c__ControlAction_message_member_array,  // message members
  ControlAction__rosidl_typesupport_introspection_c__ControlAction_init_function,  // function to initialize message memory (memory has to be allocated)
  ControlAction__rosidl_typesupport_introspection_c__ControlAction_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ControlAction__rosidl_typesupport_introspection_c__ControlAction_message_type_support_handle = {
  0,
  &ControlAction__rosidl_typesupport_introspection_c__ControlAction_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cmrdv_interfaces, msg, ControlAction)() {
  if (!ControlAction__rosidl_typesupport_introspection_c__ControlAction_message_type_support_handle.typesupport_identifier) {
    ControlAction__rosidl_typesupport_introspection_c__ControlAction_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ControlAction__rosidl_typesupport_introspection_c__ControlAction_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
