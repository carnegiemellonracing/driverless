// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from interfaces:msg/ConeList.idl
// generated code does not contain a copyright notice

#ifndef INTERFACES__MSG__DETAIL__CONE_LIST__STRUCT_H_
#define INTERFACES__MSG__DETAIL__CONE_LIST__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'blue_cones'
// Member 'yellow_cones'
// Member 'orange_cones'
#include "geometry_msgs/msg/detail/point__struct.h"

// Struct defined in msg/ConeList in the package interfaces.
typedef struct interfaces__msg__ConeList
{
  geometry_msgs__msg__Point__Sequence blue_cones;
  geometry_msgs__msg__Point__Sequence yellow_cones;
  geometry_msgs__msg__Point__Sequence orange_cones;
} interfaces__msg__ConeList;

// Struct for a sequence of interfaces__msg__ConeList.
typedef struct interfaces__msg__ConeList__Sequence
{
  interfaces__msg__ConeList * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} interfaces__msg__ConeList__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // INTERFACES__MSG__DETAIL__CONE_LIST__STRUCT_H_
