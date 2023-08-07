// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/VehicleCommands.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in msg/VehicleCommands in the package eufs_msgs.
typedef struct eufs_msgs__msg__VehicleCommands
{
  int8_t handshake;
  int8_t ebs;
  int8_t direction;
  int8_t mission_status;
  double braking;
  double torque;
  double steering;
  double rpm;
} eufs_msgs__msg__VehicleCommands;

// Struct for a sequence of eufs_msgs__msg__VehicleCommands.
typedef struct eufs_msgs__msg__VehicleCommands__Sequence
{
  eufs_msgs__msg__VehicleCommands * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__VehicleCommands__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__VEHICLE_COMMANDS__STRUCT_H_
