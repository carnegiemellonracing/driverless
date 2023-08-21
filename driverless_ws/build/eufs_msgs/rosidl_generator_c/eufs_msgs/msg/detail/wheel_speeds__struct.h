// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/WheelSpeeds.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in msg/WheelSpeeds in the package eufs_msgs.
typedef struct eufs_msgs__msg__WheelSpeeds
{
  float steering;
  float lf_speed;
  float rf_speed;
  float lb_speed;
  float rb_speed;
} eufs_msgs__msg__WheelSpeeds;

// Struct for a sequence of eufs_msgs__msg__WheelSpeeds.
typedef struct eufs_msgs__msg__WheelSpeeds__Sequence
{
  eufs_msgs__msg__WheelSpeeds * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__WheelSpeeds__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__WHEEL_SPEEDS__STRUCT_H_
