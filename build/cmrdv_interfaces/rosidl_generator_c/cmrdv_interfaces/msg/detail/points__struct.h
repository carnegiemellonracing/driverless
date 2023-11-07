// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cmrdv_interfaces:msg/Points.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__POINTS__STRUCT_H_
#define CMRDV_INTERFACES__MSG__DETAIL__POINTS__STRUCT_H_

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
// Member 'points'
#include "geometry_msgs/msg/detail/point__struct.h"

// Struct defined in msg/Points in the package cmrdv_interfaces.
typedef struct cmrdv_interfaces__msg__Points
{
  std_msgs__msg__Header header;
  geometry_msgs__msg__Point__Sequence points;
} cmrdv_interfaces__msg__Points;

// Struct for a sequence of cmrdv_interfaces__msg__Points.
typedef struct cmrdv_interfaces__msg__Points__Sequence
{
  cmrdv_interfaces__msg__Points * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cmrdv_interfaces__msg__Points__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__POINTS__STRUCT_H_
