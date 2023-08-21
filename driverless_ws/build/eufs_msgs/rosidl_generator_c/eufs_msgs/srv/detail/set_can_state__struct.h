// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:srv/SetCanState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__SRV__DETAIL__SET_CAN_STATE__STRUCT_H_
#define EUFS_MSGS__SRV__DETAIL__SET_CAN_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in srv/SetCanState in the package eufs_msgs.
typedef struct eufs_msgs__srv__SetCanState_Request
{
  uint16_t ami_state;
  uint16_t as_state;
} eufs_msgs__srv__SetCanState_Request;

// Struct for a sequence of eufs_msgs__srv__SetCanState_Request.
typedef struct eufs_msgs__srv__SetCanState_Request__Sequence
{
  eufs_msgs__srv__SetCanState_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__srv__SetCanState_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

// Struct defined in srv/SetCanState in the package eufs_msgs.
typedef struct eufs_msgs__srv__SetCanState_Response
{
  bool success;
  rosidl_runtime_c__String message;
} eufs_msgs__srv__SetCanState_Response;

// Struct for a sequence of eufs_msgs__srv__SetCanState_Response.
typedef struct eufs_msgs__srv__SetCanState_Response__Sequence
{
  eufs_msgs__srv__SetCanState_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__srv__SetCanState_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__SRV__DETAIL__SET_CAN_STATE__STRUCT_H_
