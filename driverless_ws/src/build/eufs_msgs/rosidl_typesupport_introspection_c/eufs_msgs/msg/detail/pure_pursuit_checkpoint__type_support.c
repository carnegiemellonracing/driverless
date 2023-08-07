// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/PurePursuitCheckpoint.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/pure_pursuit_checkpoint__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/pure_pursuit_checkpoint__functions.h"
#include "eufs_msgs/msg/detail/pure_pursuit_checkpoint__struct.h"


// Include directives for member types
// Member `position`
#include "geometry_msgs/msg/point.h"
// Member `position`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__PurePursuitCheckpoint__init(message_memory);
}

void PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_fini_function(void * message_memory)
{
  eufs_msgs__msg__PurePursuitCheckpoint__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_message_member_array[3] = {
  {
    "position",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PurePursuitCheckpoint, position),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "max_speed",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PurePursuitCheckpoint, max_speed),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "max_lateral_acceleration",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PurePursuitCheckpoint, max_lateral_acceleration),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_message_members = {
  "eufs_msgs__msg",  // message namespace
  "PurePursuitCheckpoint",  // message name
  3,  // number of fields
  sizeof(eufs_msgs__msg__PurePursuitCheckpoint),
  PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_message_member_array,  // message members
  PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_init_function,  // function to initialize message memory (memory has to be allocated)
  PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_message_type_support_handle = {
  0,
  &PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, PurePursuitCheckpoint)() {
  PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_message_type_support_handle.typesupport_identifier) {
    PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &PurePursuitCheckpoint__rosidl_typesupport_introspection_c__PurePursuitCheckpoint_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
