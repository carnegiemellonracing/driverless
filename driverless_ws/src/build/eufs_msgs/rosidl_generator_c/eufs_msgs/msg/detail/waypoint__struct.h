// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/Waypoint.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WAYPOINT__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__WAYPOINT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'position'
#include "geometry_msgs/msg/detail/point__struct.h"

// Struct defined in msg/Waypoint in the package eufs_msgs.
typedef struct eufs_msgs__msg__Waypoint
{
  geometry_msgs__msg__Point position;
  double speed;
  double suggested_steering;
} eufs_msgs__msg__Waypoint;

// Struct for a sequence of eufs_msgs__msg__Waypoint.
typedef struct eufs_msgs__msg__Waypoint__Sequence
{
  eufs_msgs__msg__Waypoint * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__Waypoint__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__WAYPOINT__STRUCT_H_
