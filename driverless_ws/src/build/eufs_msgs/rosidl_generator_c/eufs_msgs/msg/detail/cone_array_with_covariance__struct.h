// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/ConeArrayWithCovariance.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__STRUCT_H_

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
// Member 'blue_cones'
// Member 'yellow_cones'
// Member 'orange_cones'
// Member 'big_orange_cones'
// Member 'unknown_color_cones'
#include "eufs_msgs/msg/detail/cone_with_covariance__struct.h"

// Struct defined in msg/ConeArrayWithCovariance in the package eufs_msgs.
typedef struct eufs_msgs__msg__ConeArrayWithCovariance
{
  std_msgs__msg__Header header;
  eufs_msgs__msg__ConeWithCovariance__Sequence blue_cones;
  eufs_msgs__msg__ConeWithCovariance__Sequence yellow_cones;
  eufs_msgs__msg__ConeWithCovariance__Sequence orange_cones;
  eufs_msgs__msg__ConeWithCovariance__Sequence big_orange_cones;
  eufs_msgs__msg__ConeWithCovariance__Sequence unknown_color_cones;
} eufs_msgs__msg__ConeArrayWithCovariance;

// Struct for a sequence of eufs_msgs__msg__ConeArrayWithCovariance.
typedef struct eufs_msgs__msg__ConeArrayWithCovariance__Sequence
{
  eufs_msgs__msg__ConeArrayWithCovariance * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__ConeArrayWithCovariance__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__STRUCT_H_
