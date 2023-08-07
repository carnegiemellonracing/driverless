// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/Heartbeat.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__HEARTBEAT__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__HEARTBEAT__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in msg/Heartbeat in the package eufs_msgs.
typedef struct eufs_msgs__msg__Heartbeat
{
  uint8_t id;
  uint8_t data;
} eufs_msgs__msg__Heartbeat;

// Struct for a sequence of eufs_msgs__msg__Heartbeat.
typedef struct eufs_msgs__msg__Heartbeat__Sequence
{
  eufs_msgs__msg__Heartbeat * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__Heartbeat__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__HEARTBEAT__STRUCT_H_
