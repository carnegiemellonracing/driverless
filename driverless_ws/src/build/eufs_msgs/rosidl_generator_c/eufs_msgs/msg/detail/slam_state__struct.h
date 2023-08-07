// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/SLAMState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__SLAM_STATE__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__SLAM_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'NO_INPUTS'.
enum
{
  eufs_msgs__msg__SLAMState__NO_INPUTS = 0
};

/// Constant 'MAPPING'.
enum
{
  eufs_msgs__msg__SLAMState__MAPPING = 1
};

/// Constant 'LOCALISING'.
enum
{
  eufs_msgs__msg__SLAMState__LOCALISING = 2
};

/// Constant 'TIMEOUT'.
enum
{
  eufs_msgs__msg__SLAMState__TIMEOUT = 3
};

/// Constant 'RECOMMENDS_FAILURE'.
enum
{
  eufs_msgs__msg__SLAMState__RECOMMENDS_FAILURE = 4
};

// Include directives for member types
// Member 'status'
#include "rosidl_runtime_c/string.h"

// Struct defined in msg/SLAMState in the package eufs_msgs.
typedef struct eufs_msgs__msg__SLAMState
{
  bool loop_closed;
  int16_t laps;
  rosidl_runtime_c__String status;
  int8_t state;
} eufs_msgs__msg__SLAMState;

// Struct for a sequence of eufs_msgs__msg__SLAMState.
typedef struct eufs_msgs__msg__SLAMState__Sequence
{
  eufs_msgs__msg__SLAMState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__SLAMState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__SLAM_STATE__STRUCT_H_
