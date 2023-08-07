// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/SystemStatus.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__SYSTEM_STATUS__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__SYSTEM_STATUS__STRUCT_H_

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
// Member 'topic_statuses'
#include "eufs_msgs/msg/detail/topic_status__struct.h"

// Struct defined in msg/SystemStatus in the package eufs_msgs.
typedef struct eufs_msgs__msg__SystemStatus
{
  std_msgs__msg__Header header;
  eufs_msgs__msg__TopicStatus__Sequence topic_statuses;
} eufs_msgs__msg__SystemStatus;

// Struct for a sequence of eufs_msgs__msg__SystemStatus.
typedef struct eufs_msgs__msg__SystemStatus__Sequence
{
  eufs_msgs__msg__SystemStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__SystemStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__SYSTEM_STATUS__STRUCT_H_
