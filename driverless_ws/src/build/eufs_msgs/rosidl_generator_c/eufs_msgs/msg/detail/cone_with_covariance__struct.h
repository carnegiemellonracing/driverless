// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/ConeWithCovariance.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CONE_WITH_COVARIANCE__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__CONE_WITH_COVARIANCE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'point'
#include "geometry_msgs/msg/detail/point__struct.h"

// Struct defined in msg/ConeWithCovariance in the package eufs_msgs.
typedef struct eufs_msgs__msg__ConeWithCovariance
{
  geometry_msgs__msg__Point point;
  double covariance[4];
} eufs_msgs__msg__ConeWithCovariance;

// Struct for a sequence of eufs_msgs__msg__ConeWithCovariance.
typedef struct eufs_msgs__msg__ConeWithCovariance__Sequence
{
  eufs_msgs__msg__ConeWithCovariance * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__ConeWithCovariance__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__CONE_WITH_COVARIANCE__STRUCT_H_
