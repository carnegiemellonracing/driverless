// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/ConeArray.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CONE_ARRAY__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__CONE_ARRAY__STRUCT_H_

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
#include "geometry_msgs/msg/detail/point__struct.h"

// Struct defined in msg/ConeArray in the package eufs_msgs.
typedef struct eufs_msgs__msg__ConeArray
{
  std_msgs__msg__Header header;
  geometry_msgs__msg__Point__Sequence blue_cones;
  geometry_msgs__msg__Point__Sequence yellow_cones;
  geometry_msgs__msg__Point__Sequence orange_cones;
  geometry_msgs__msg__Point__Sequence big_orange_cones;
  geometry_msgs__msg__Point__Sequence unknown_color_cones;
} eufs_msgs__msg__ConeArray;

// Struct for a sequence of eufs_msgs__msg__ConeArray.
typedef struct eufs_msgs__msg__ConeArray__Sequence
{
  eufs_msgs__msg__ConeArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__ConeArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__CONE_ARRAY__STRUCT_H_
