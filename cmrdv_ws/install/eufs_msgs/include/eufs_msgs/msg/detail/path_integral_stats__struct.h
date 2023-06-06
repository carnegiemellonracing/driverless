// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/PathIntegralStats.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__STRUCT_H_

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
// Member 'tag'
#include "rosidl_runtime_c/string.h"
// Member 'params'
#include "eufs_msgs/msg/detail/path_integral_params__struct.h"
// Member 'stats'
#include "eufs_msgs/msg/detail/lap_stats__struct.h"

// Struct defined in msg/PathIntegralStats in the package eufs_msgs.
typedef struct eufs_msgs__msg__PathIntegralStats
{
  std_msgs__msg__Header header;
  rosidl_runtime_c__String tag;
  eufs_msgs__msg__PathIntegralParams params;
  eufs_msgs__msg__LapStats stats;
} eufs_msgs__msg__PathIntegralStats;

// Struct for a sequence of eufs_msgs__msg__PathIntegralStats.
typedef struct eufs_msgs__msg__PathIntegralStats__Sequence
{
  eufs_msgs__msg__PathIntegralStats * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__PathIntegralStats__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__STRUCT_H_
