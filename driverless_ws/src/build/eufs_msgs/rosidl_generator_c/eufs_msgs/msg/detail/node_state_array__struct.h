// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/NodeStateArray.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__NODE_STATE_ARRAY__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__NODE_STATE_ARRAY__STRUCT_H_

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
// Member 'states'
#include "eufs_msgs/msg/detail/node_state__struct.h"

// Struct defined in msg/NodeStateArray in the package eufs_msgs.
typedef struct eufs_msgs__msg__NodeStateArray
{
  std_msgs__msg__Header header;
  eufs_msgs__msg__NodeState__Sequence states;
} eufs_msgs__msg__NodeStateArray;

// Struct for a sequence of eufs_msgs__msg__NodeStateArray.
typedef struct eufs_msgs__msg__NodeStateArray__Sequence
{
  eufs_msgs__msg__NodeStateArray * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__NodeStateArray__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__NODE_STATE_ARRAY__STRUCT_H_
