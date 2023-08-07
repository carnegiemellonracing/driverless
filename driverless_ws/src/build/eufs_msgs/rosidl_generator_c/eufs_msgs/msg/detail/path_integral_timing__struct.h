// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/PathIntegralTiming.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_TIMING__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_TIMING__STRUCT_H_

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

// Struct defined in msg/PathIntegralTiming in the package eufs_msgs.
typedef struct eufs_msgs__msg__PathIntegralTiming
{
  std_msgs__msg__Header header;
  double average_time_between_poses;
  double average_optimization_cycle_time;
  double average_sleep_time;
} eufs_msgs__msg__PathIntegralTiming;

// Struct for a sequence of eufs_msgs__msg__PathIntegralTiming.
typedef struct eufs_msgs__msg__PathIntegralTiming__Sequence
{
  eufs_msgs__msg__PathIntegralTiming * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__PathIntegralTiming__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_TIMING__STRUCT_H_
