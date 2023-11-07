// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cmrdv_interfaces:msg/VehicleState.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__VEHICLE_STATE__STRUCT_H_
#define CMRDV_INTERFACES__MSG__DETAIL__VEHICLE_STATE__STRUCT_H_

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
// Member 'position'
#include "geometry_msgs/msg/detail/pose__struct.h"
// Member 'velocity'
#include "geometry_msgs/msg/detail/twist__struct.h"
// Member 'acceleration'
#include "geometry_msgs/msg/detail/accel__struct.h"

// Struct defined in msg/VehicleState in the package cmrdv_interfaces.
typedef struct cmrdv_interfaces__msg__VehicleState
{
  std_msgs__msg__Header header;
  geometry_msgs__msg__Pose position;
  geometry_msgs__msg__Twist velocity;
  geometry_msgs__msg__Accel acceleration;
} cmrdv_interfaces__msg__VehicleState;

// Struct for a sequence of cmrdv_interfaces__msg__VehicleState.
typedef struct cmrdv_interfaces__msg__VehicleState__Sequence
{
  cmrdv_interfaces__msg__VehicleState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cmrdv_interfaces__msg__VehicleState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__VEHICLE_STATE__STRUCT_H_
