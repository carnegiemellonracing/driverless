// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cmrdv_interfaces:msg/Brakes.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__BRAKES__STRUCT_H_
#define CMRDV_INTERFACES__MSG__DETAIL__BRAKES__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'last_fired'
#include "builtin_interfaces/msg/detail/time__struct.h"

// Struct defined in msg/Brakes in the package cmrdv_interfaces.
typedef struct cmrdv_interfaces__msg__Brakes
{
  bool braking;
  builtin_interfaces__msg__Time last_fired;
} cmrdv_interfaces__msg__Brakes;

// Struct for a sequence of cmrdv_interfaces__msg__Brakes.
typedef struct cmrdv_interfaces__msg__Brakes__Sequence
{
  cmrdv_interfaces__msg__Brakes * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cmrdv_interfaces__msg__Brakes__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__BRAKES__STRUCT_H_
