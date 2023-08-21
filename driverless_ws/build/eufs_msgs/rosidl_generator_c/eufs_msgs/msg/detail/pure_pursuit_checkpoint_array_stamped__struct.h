// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/PurePursuitCheckpointArrayStamped.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT_ARRAY_STAMPED__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT_ARRAY_STAMPED__STRUCT_H_

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
// Member 'checkpoints'
#include "eufs_msgs/msg/detail/pure_pursuit_checkpoint__struct.h"

// Struct defined in msg/PurePursuitCheckpointArrayStamped in the package eufs_msgs.
typedef struct eufs_msgs__msg__PurePursuitCheckpointArrayStamped
{
  std_msgs__msg__Header header;
  eufs_msgs__msg__PurePursuitCheckpoint__Sequence checkpoints;
} eufs_msgs__msg__PurePursuitCheckpointArrayStamped;

// Struct for a sequence of eufs_msgs__msg__PurePursuitCheckpointArrayStamped.
typedef struct eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence
{
  eufs_msgs__msg__PurePursuitCheckpointArrayStamped * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__PURE_PURSUIT_CHECKPOINT_ARRAY_STAMPED__STRUCT_H_
