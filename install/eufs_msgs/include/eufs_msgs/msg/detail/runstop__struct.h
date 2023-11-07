// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/Runstop.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__RUNSTOP__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__RUNSTOP__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'sender'
#include "rosidl_runtime_c/string.h"

// Struct defined in msg/Runstop in the package eufs_msgs.
typedef struct eufs_msgs__msg__Runstop
{
  std_msgs__msg__Header header;
  rosidl_runtime_c__String sender;
  bool motion_enabled;
} eufs_msgs__msg__Runstop;

// Struct for a sequence of eufs_msgs__msg__Runstop.
typedef struct eufs_msgs__msg__Runstop__Sequence
{
  eufs_msgs__msg__Runstop * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__Runstop__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__RUNSTOP__STRUCT_H_
