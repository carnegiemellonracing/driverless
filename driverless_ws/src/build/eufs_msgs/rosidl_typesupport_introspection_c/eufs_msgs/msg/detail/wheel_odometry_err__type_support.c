// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/WheelOdometryErr.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/wheel_odometry_err__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/wheel_odometry_err__functions.h"
#include "eufs_msgs/msg/detail/wheel_odometry_err__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__WheelOdometryErr__init(message_memory);
}

void WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_fini_function(void * message_memory)
{
  eufs_msgs__msg__WheelOdometryErr__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_message_member_array[6] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WheelOdometryErr, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "position_err",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WheelOdometryErr, position_err),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "orientation_err",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WheelOdometryErr, orientation_err),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "linear_vel_err",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WheelOdometryErr, linear_vel_err),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "angular_vel_err",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WheelOdometryErr, angular_vel_err),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "forward_vel_err",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WheelOdometryErr, forward_vel_err),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_message_members = {
  "eufs_msgs__msg",  // message namespace
  "WheelOdometryErr",  // message name
  6,  // number of fields
  sizeof(eufs_msgs__msg__WheelOdometryErr),
  WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_message_member_array,  // message members
  WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_init_function,  // function to initialize message memory (memory has to be allocated)
  WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_message_type_support_handle = {
  0,
  &WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, WheelOdometryErr)() {
  WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_message_type_support_handle.typesupport_identifier) {
    WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &WheelOdometryErr__rosidl_typesupport_introspection_c__WheelOdometryErr_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
