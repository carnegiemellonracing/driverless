// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/PurePursuitCheckpoint.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__STRUCT_H_

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

// Struct defined in msg/PurePursuitCheckpoint in the package eufs_msgs.
typedef struct eufs_msgs__msg__PurePursuitCheckpoint
{
  geometry_msgs__msg__Point position;
  double max_speed;
  double max_lateral_acceleration;
} eufs_msgs__msg__PurePursuitCheckpoint;

// Struct for a sequence of eufs_msgs__msg__PurePursuitCheckpoint.
typedef struct eufs_msgs__msg__PurePursuitCheckpoint__Sequence
{
  eufs_msgs__msg__PurePursuitCheckpoint * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__PurePursuitCheckpoint__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT__STRUCT_H_
