// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/StateMachineState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__STATE_MACHINE_STATE__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__STATE_MACHINE_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'state_str'
#include "rosidl_runtime_c/string.h"

// Struct defined in msg/StateMachineState in the package eufs_msgs.
typedef struct eufs_msgs__msg__StateMachineState
{
  uint16_t state;
  rosidl_runtime_c__String state_str;
} eufs_msgs__msg__StateMachineState;

// Struct for a sequence of eufs_msgs__msg__StateMachineState.
typedef struct eufs_msgs__msg__StateMachineState__Sequence
{
  eufs_msgs__msg__StateMachineState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__StateMachineState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__STATE_MACHINE_STATE__STRUCT_H_
