// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/PlanningMode.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'LOCAL'.
enum
{
  eufs_msgs__msg__PlanningMode__LOCAL = 0
};

/// Constant 'GLOBAL'.
enum
{
  eufs_msgs__msg__PlanningMode__GLOBAL = 1
};

// Struct defined in msg/PlanningMode in the package eufs_msgs.
typedef struct eufs_msgs__msg__PlanningMode
{
  int8_t mode;
} eufs_msgs__msg__PlanningMode;

// Struct for a sequence of eufs_msgs__msg__PlanningMode.
typedef struct eufs_msgs__msg__PlanningMode__Sequence
{
  eufs_msgs__msg__PlanningMode * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__PlanningMode__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__PLANNING_MODE__STRUCT_H_
