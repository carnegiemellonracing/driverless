// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/MPCState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__MPC_STATE__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__MPC_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in msg/MPCState in the package eufs_msgs.
typedef struct eufs_msgs__msg__MPCState
{
  int8_t exitflag;
  uint8_t iterations;
  double solve_time;
  double cost;
} eufs_msgs__msg__MPCState;

// Struct for a sequence of eufs_msgs__msg__MPCState.
typedef struct eufs_msgs__msg__MPCState__Sequence
{
  eufs_msgs__msg__MPCState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__MPCState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__MPC_STATE__STRUCT_H_
