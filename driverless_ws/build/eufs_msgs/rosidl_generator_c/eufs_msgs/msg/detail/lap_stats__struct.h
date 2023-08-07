// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/LapStats.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__LAP_STATS__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__LAP_STATS__STRUCT_H_

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

// Struct defined in msg/LapStats in the package eufs_msgs.
typedef struct eufs_msgs__msg__LapStats
{
  std_msgs__msg__Header header;
  int64_t lap_number;
  double lap_time;
  double avg_speed;
  double max_speed;
  double speed_var;
  double max_slip;
  double max_lateral_accel;
  double normalized_deviation_mse;
  double deviation_var;
  double max_deviation;
} eufs_msgs__msg__LapStats;

// Struct for a sequence of eufs_msgs__msg__LapStats.
typedef struct eufs_msgs__msg__LapStats__Sequence
{
  eufs_msgs__msg__LapStats * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__LapStats__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__LAP_STATS__STRUCT_H_
