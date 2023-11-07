// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cmrdv_interfaces:msg/CarROT.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__STRUCT_H_
#define CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__STRUCT_H_

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

// Struct defined in msg/CarROT in the package cmrdv_interfaces.
typedef struct cmrdv_interfaces__msg__CarROT
{
  std_msgs__msg__Header header;
  double x;
  double y;
  double yaw;
  double curvature;
} cmrdv_interfaces__msg__CarROT;

// Struct for a sequence of cmrdv_interfaces__msg__CarROT.
typedef struct cmrdv_interfaces__msg__CarROT__Sequence
{
  cmrdv_interfaces__msg__CarROT * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cmrdv_interfaces__msg__CarROT__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__STRUCT_H_
