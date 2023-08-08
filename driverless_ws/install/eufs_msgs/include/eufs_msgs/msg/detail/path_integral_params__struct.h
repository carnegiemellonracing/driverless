// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/PathIntegralParams.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'map_path'
#include "rosidl_runtime_c/string.h"

// Struct defined in msg/PathIntegralParams in the package eufs_msgs.
typedef struct eufs_msgs__msg__PathIntegralParams
{
  int64_t hz;
  int64_t num_timesteps;
  int64_t num_iters;
  double gamma;
  double init_steering;
  double init_throttle;
  double steering_var;
  double throttle_var;
  double max_throttle;
  double speed_coefficient;
  double track_coefficient;
  double max_slip_angle;
  double track_slop;
  double crash_coeff;
  rosidl_runtime_c__String map_path;
  double desired_speed;
} eufs_msgs__msg__PathIntegralParams;

// Struct for a sequence of eufs_msgs__msg__PathIntegralParams.
typedef struct eufs_msgs__msg__PathIntegralParams__Sequence
{
  eufs_msgs__msg__PathIntegralParams * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__PathIntegralParams__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__STRUCT_H_
