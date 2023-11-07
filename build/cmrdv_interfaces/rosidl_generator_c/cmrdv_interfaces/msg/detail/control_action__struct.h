// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cmrdv_interfaces:msg/ControlAction.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__STRUCT_H_
#define CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in msg/ControlAction in the package cmrdv_interfaces.
typedef struct cmrdv_interfaces__msg__ControlAction
{
  double wheel_speed;
  double swangle;
} cmrdv_interfaces__msg__ControlAction;

// Struct for a sequence of cmrdv_interfaces__msg__ControlAction.
typedef struct cmrdv_interfaces__msg__ControlAction__Sequence
{
  cmrdv_interfaces__msg__ControlAction * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cmrdv_interfaces__msg__ControlAction__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CONTROL_ACTION__STRUCT_H_
