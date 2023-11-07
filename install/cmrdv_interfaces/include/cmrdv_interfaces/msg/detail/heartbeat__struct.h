// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cmrdv_interfaces:msg/Heartbeat.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__HEARTBEAT__STRUCT_H_
#define CMRDV_INTERFACES__MSG__DETAIL__HEARTBEAT__STRUCT_H_

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

// Struct defined in msg/Heartbeat in the package cmrdv_interfaces.
typedef struct cmrdv_interfaces__msg__Heartbeat
{
  std_msgs__msg__Header header;
  uint16_t status;
} cmrdv_interfaces__msg__Heartbeat;

// Struct for a sequence of cmrdv_interfaces__msg__Heartbeat.
typedef struct cmrdv_interfaces__msg__Heartbeat__Sequence
{
  cmrdv_interfaces__msg__Heartbeat * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cmrdv_interfaces__msg__Heartbeat__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__HEARTBEAT__STRUCT_H_
