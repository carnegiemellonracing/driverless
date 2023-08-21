// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/IntegrationErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__INTEGRATION_ERR__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__INTEGRATION_ERR__STRUCT_H_

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

// Struct defined in msg/IntegrationErr in the package eufs_msgs.
typedef struct eufs_msgs__msg__IntegrationErr
{
  std_msgs__msg__Header header;
  double position_err;
  double orientation_err;
  double linear_vel_err;
  double angular_vel_err;
} eufs_msgs__msg__IntegrationErr;

// Struct for a sequence of eufs_msgs__msg__IntegrationErr.
typedef struct eufs_msgs__msg__IntegrationErr__Sequence
{
  eufs_msgs__msg__IntegrationErr * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__IntegrationErr__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__INTEGRATION_ERR__STRUCT_H_
