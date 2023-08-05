// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from interfaces:msg/PairROT.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "interfaces/msg/detail/pair_rot__rosidl_typesupport_introspection_c.h"
#include "interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "interfaces/msg/detail/pair_rot__functions.h"
#include "interfaces/msg/detail/pair_rot__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `near`
// Member `far`
#include "interfaces/msg/car_rot.h"
// Member `near`
// Member `far`
#include "interfaces/msg/detail/car_rot__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  interfaces__msg__PairROT__init(message_memory);
}

void interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_fini_function(void * message_memory)
{
  interfaces__msg__PairROT__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_message_member_array[3] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(interfaces__msg__PairROT, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "near",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(interfaces__msg__PairROT, near),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "far",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(interfaces__msg__PairROT, far),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_message_members = {
  "interfaces__msg",  // message namespace
  "PairROT",  // message name
  3,  // number of fields
  sizeof(interfaces__msg__PairROT),
  interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_message_member_array,  // message members
  interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_init_function,  // function to initialize message memory (memory has to be allocated)
  interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_message_type_support_handle = {
  0,
  &interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_message_members,
  get_message_typesupport_handle_function,
  &interfaces__msg__PairROT__get_type_hash,
  &interfaces__msg__PairROT__get_type_description,
  &interfaces__msg__PairROT__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, interfaces, msg, PairROT)() {
  interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, interfaces, msg, CarROT)();
  interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, interfaces, msg, CarROT)();
  if (!interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_message_type_support_handle.typesupport_identifier) {
    interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &interfaces__msg__PairROT__rosidl_typesupport_introspection_c__PairROT_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
