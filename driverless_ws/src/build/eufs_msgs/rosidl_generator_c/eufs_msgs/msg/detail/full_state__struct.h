// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/FullState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__FULL_STATE__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__FULL_STATE__STRUCT_H_

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

// Struct defined in msg/FullState in the package eufs_msgs.
typedef struct eufs_msgs__msg__FullState
{
  std_msgs__msg__Header header;
  double x_pos;
  double y_pos;
  double yaw;
  double roll;
  double u_x;
  double u_y;
  double yaw_mder;
  double front_throttle;
  double rear_throttle;
  double steering;
} eufs_msgs__msg__FullState;

// Struct for a sequence of eufs_msgs__msg__FullState.
typedef struct eufs_msgs__msg__FullState__Sequence
{
  eufs_msgs__msg__FullState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__FullState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__FULL_STATE__STRUCT_H_
