// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/Costmap.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__COSTMAP__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__COSTMAP__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'channel0'
// Member 'channel1'
// Member 'channel2'
// Member 'channel3'
#include "rosidl_runtime_c/primitives_sequence.h"

// Struct defined in msg/Costmap in the package eufs_msgs.
typedef struct eufs_msgs__msg__Costmap
{
  double x_bounds_min;
  double x_bounds_max;
  double y_bounds_min;
  double y_bounds_max;
  double pixels_per_meter;
  rosidl_runtime_c__float__Sequence channel0;
  rosidl_runtime_c__float__Sequence channel1;
  rosidl_runtime_c__float__Sequence channel2;
  rosidl_runtime_c__float__Sequence channel3;
} eufs_msgs__msg__Costmap;

// Struct for a sequence of eufs_msgs__msg__Costmap.
typedef struct eufs_msgs__msg__Costmap__Sequence
{
  eufs_msgs__msg__Costmap * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__Costmap__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__COSTMAP__STRUCT_H_
