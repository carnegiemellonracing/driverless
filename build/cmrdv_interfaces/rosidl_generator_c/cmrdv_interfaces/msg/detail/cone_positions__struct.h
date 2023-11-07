// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cmrdv_interfaces:msg/ConePositions.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__STRUCT_H_
#define CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__STRUCT_H_

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
// Member 'cone_positions'
#include "std_msgs/msg/detail/float32__struct.h"

// Struct defined in msg/ConePositions in the package cmrdv_interfaces.
typedef struct cmrdv_interfaces__msg__ConePositions
{
  std_msgs__msg__Header header;
  std_msgs__msg__Float32__Sequence cone_positions;
} cmrdv_interfaces__msg__ConePositions;

// Struct for a sequence of cmrdv_interfaces__msg__ConePositions.
typedef struct cmrdv_interfaces__msg__ConePositions__Sequence
{
  cmrdv_interfaces__msg__ConePositions * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cmrdv_interfaces__msg__ConePositions__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__STRUCT_H_
