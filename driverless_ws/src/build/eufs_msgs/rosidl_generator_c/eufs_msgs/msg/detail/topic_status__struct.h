// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/TopicStatus.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'OFF'.
enum
{
  eufs_msgs__msg__TopicStatus__OFF = 0
};

/// Constant 'PUBLISHING'.
enum
{
  eufs_msgs__msg__TopicStatus__PUBLISHING = 1
};

/// Constant 'TIMEOUT_EXCEEDED'.
enum
{
  eufs_msgs__msg__TopicStatus__TIMEOUT_EXCEEDED = 2
};

// Include directives for member types
// Member 'topic'
// Member 'description'
// Member 'group'
// Member 'log_level'
#include "rosidl_runtime_c/string.h"

// Struct defined in msg/TopicStatus in the package eufs_msgs.
typedef struct eufs_msgs__msg__TopicStatus
{
  rosidl_runtime_c__String topic;
  rosidl_runtime_c__String description;
  rosidl_runtime_c__String group;
  bool trigger_ebs;
  rosidl_runtime_c__String log_level;
  uint16_t status;
} eufs_msgs__msg__TopicStatus;

// Struct for a sequence of eufs_msgs__msg__TopicStatus.
typedef struct eufs_msgs__msg__TopicStatus__Sequence
{
  eufs_msgs__msg__TopicStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__TopicStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__STRUCT_H_
