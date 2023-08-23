// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from interfaces:msg/PairROT.idl
// generated code does not contain a copyright notice

#ifndef INTERFACES__MSG__DETAIL__PAIR_ROT__STRUCT_H_
#define INTERFACES__MSG__DETAIL__PAIR_ROT__STRUCT_H_

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
// Member 'near'
// Member 'far'
#include "interfaces/msg/detail/car_rot__struct.h"

// Struct defined in msg/PairROT in the package interfaces.
typedef struct interfaces__msg__PairROT
{
  std_msgs__msg__Header header;
  interfaces__msg__CarROT near;
  interfaces__msg__CarROT far;
} interfaces__msg__PairROT;

// Struct for a sequence of interfaces__msg__PairROT.
typedef struct interfaces__msg__PairROT__Sequence
{
  interfaces__msg__PairROT * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} interfaces__msg__PairROT__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // INTERFACES__MSG__DETAIL__PAIR_ROT__STRUCT_H_
