// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/BoundingBox.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__BOUNDING_BOX__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__BOUNDING_BOX__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'PIXEL'.
enum
{
  eufs_msgs__msg__BoundingBox__PIXEL = 0l
};

/// Constant 'PERCENTAGE'.
enum
{
  eufs_msgs__msg__BoundingBox__PERCENTAGE = 1l
};

// Include directives for member types
// Member 'color'
#include "rosidl_runtime_c/string.h"

// Struct defined in msg/BoundingBox in the package eufs_msgs.
typedef struct eufs_msgs__msg__BoundingBox
{
  rosidl_runtime_c__String color;
  double probability;
  int32_t type;
  double xmin;
  double ymin;
  double xmax;
  double ymax;
} eufs_msgs__msg__BoundingBox;

// Struct for a sequence of eufs_msgs__msg__BoundingBox.
typedef struct eufs_msgs__msg__BoundingBox__Sequence
{
  eufs_msgs__msg__BoundingBox * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__BoundingBox__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__BOUNDING_BOX__STRUCT_H_
