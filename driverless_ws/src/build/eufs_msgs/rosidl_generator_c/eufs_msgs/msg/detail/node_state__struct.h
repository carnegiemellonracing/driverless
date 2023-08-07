// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/NodeState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__NODE_STATE__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__NODE_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'name'
#include "rosidl_runtime_c/string.h"
// Member 'last_heartbeat'
#include "builtin_interfaces/msg/detail/time__struct.h"

// Struct defined in msg/NodeState in the package eufs_msgs.
typedef struct eufs_msgs__msg__NodeState
{
  uint16_t id;
  rosidl_runtime_c__String name;
  uint8_t exp_heartbeat;
  builtin_interfaces__msg__Time last_heartbeat;
  uint8_t severity;
  bool online;
} eufs_msgs__msg__NodeState;

// Struct for a sequence of eufs_msgs__msg__NodeState.
typedef struct eufs_msgs__msg__NodeState__Sequence
{
  eufs_msgs__msg__NodeState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__NodeState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__NODE_STATE__STRUCT_H_
