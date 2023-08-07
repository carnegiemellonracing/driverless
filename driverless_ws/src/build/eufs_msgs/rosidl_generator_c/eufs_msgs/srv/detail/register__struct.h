// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:srv/Register.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__SRV__DETAIL__REGISTER__STRUCT_H_
#define EUFS_MSGS__SRV__DETAIL__REGISTER__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'node_name'
#include "rosidl_runtime_c/string.h"

// Struct defined in srv/Register in the package eufs_msgs.
typedef struct eufs_msgs__srv__Register_Request
{
  rosidl_runtime_c__String node_name;
  uint8_t severity;
} eufs_msgs__srv__Register_Request;

// Struct for a sequence of eufs_msgs__srv__Register_Request.
typedef struct eufs_msgs__srv__Register_Request__Sequence
{
  eufs_msgs__srv__Register_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__srv__Register_Request__Sequence;


// Constants defined in the message

// Struct defined in srv/Register in the package eufs_msgs.
typedef struct eufs_msgs__srv__Register_Response
{
  uint8_t id;
} eufs_msgs__srv__Register_Response;

// Struct for a sequence of eufs_msgs__srv__Register_Response.
typedef struct eufs_msgs__srv__Register_Response__Sequence
{
  eufs_msgs__srv__Register_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__srv__Register_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__SRV__DETAIL__REGISTER__STRUCT_H_
