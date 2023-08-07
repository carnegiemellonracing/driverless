// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/SLAMErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__SLAM_ERR__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__SLAM_ERR__STRUCT_H_

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

// Struct defined in msg/SLAMErr in the package eufs_msgs.
typedef struct eufs_msgs__msg__SLAMErr
{
  std_msgs__msg__Header header;
  double x_err;
  double y_err;
  double z_err;
  double x_orient_err;
  double y_orient_err;
  double z_orient_err;
  double w_orient_err;
  double map_similarity;
} eufs_msgs__msg__SLAMErr;

// Struct for a sequence of eufs_msgs__msg__SLAMErr.
typedef struct eufs_msgs__msg__SLAMErr__Sequence
{
  eufs_msgs__msg__SLAMErr * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__SLAMErr__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__SLAM_ERR__STRUCT_H_
