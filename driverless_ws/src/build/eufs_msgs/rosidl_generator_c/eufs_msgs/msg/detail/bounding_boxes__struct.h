// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/BoundingBoxes.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__BOUNDING_BOXES__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__BOUNDING_BOXES__STRUCT_H_

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
// Member 'image_header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'bounding_boxes'
#include "eufs_msgs/msg/detail/bounding_box__struct.h"

// Struct defined in msg/BoundingBoxes in the package eufs_msgs.
typedef struct eufs_msgs__msg__BoundingBoxes
{
  std_msgs__msg__Header header;
  std_msgs__msg__Header image_header;
  eufs_msgs__msg__BoundingBox__Sequence bounding_boxes;
} eufs_msgs__msg__BoundingBoxes;

// Struct for a sequence of eufs_msgs__msg__BoundingBoxes.
typedef struct eufs_msgs__msg__BoundingBoxes__Sequence
{
  eufs_msgs__msg__BoundingBoxes * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__BoundingBoxes__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__BOUNDING_BOXES__STRUCT_H_
