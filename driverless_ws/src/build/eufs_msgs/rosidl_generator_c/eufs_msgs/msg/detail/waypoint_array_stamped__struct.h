// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/WaypointArrayStamped.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WAYPOINT_ARRAY_STAMPED__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__WAYPOINT_ARRAY_STAMPED__STRUCT_H_

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
// Member 'waypoints'
#include "eufs_msgs/msg/detail/waypoint__struct.h"

// Struct defined in msg/WaypointArrayStamped in the package eufs_msgs.
typedef struct eufs_msgs__msg__WaypointArrayStamped
{
  std_msgs__msg__Header header;
  eufs_msgs__msg__Waypoint__Sequence waypoints;
} eufs_msgs__msg__WaypointArrayStamped;

// Struct for a sequence of eufs_msgs__msg__WaypointArrayStamped.
typedef struct eufs_msgs__msg__WaypointArrayStamped__Sequence
{
  eufs_msgs__msg__WaypointArrayStamped * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__WaypointArrayStamped__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__WAYPOINT_ARRAY_STAMPED__STRUCT_H_
