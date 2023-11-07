// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cmrdv_interfaces:msg/Brakes.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cmrdv_interfaces/msg/detail/brakes__rosidl_typesupport_introspection_c.h"
#include "cmrdv_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cmrdv_interfaces/msg/detail/brakes__functions.h"
#include "cmrdv_interfaces/msg/detail/brakes__struct.h"


// Include directives for member types
// Member `last_fired`
#include "builtin_interfaces/msg/time.h"
// Member `last_fired`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void Brakes__rosidl_typesupport_introspection_c__Brakes_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cmrdv_interfaces__msg__Brakes__init(message_memory);
}

void Brakes__rosidl_typesupport_introspection_c__Brakes_fini_function(void * message_memory)
{
  cmrdv_interfaces__msg__Brakes__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember Brakes__rosidl_typesupport_introspection_c__Brakes_message_member_array[2] = {
  {
    "braking",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__Brakes, braking),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "last_fired",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__Brakes, last_fired),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers Brakes__rosidl_typesupport_introspection_c__Brakes_message_members = {
  "cmrdv_interfaces__msg",  // message namespace
  "Brakes",  // message name
  2,  // number of fields
  sizeof(cmrdv_interfaces__msg__Brakes),
  Brakes__rosidl_typesupport_introspection_c__Brakes_message_member_array,  // message members
  Brakes__rosidl_typesupport_introspection_c__Brakes_init_function,  // function to initialize message memory (memory has to be allocated)
  Brakes__rosidl_typesupport_introspection_c__Brakes_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t Brakes__rosidl_typesupport_introspection_c__Brakes_message_type_support_handle = {
  0,
  &Brakes__rosidl_typesupport_introspection_c__Brakes_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cmrdv_interfaces, msg, Brakes)() {
  Brakes__rosidl_typesupport_introspection_c__Brakes_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  if (!Brakes__rosidl_typesupport_introspection_c__Brakes_message_type_support_handle.typesupport_identifier) {
    Brakes__rosidl_typesupport_introspection_c__Brakes_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &Brakes__rosidl_typesupport_introspection_c__Brakes_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
