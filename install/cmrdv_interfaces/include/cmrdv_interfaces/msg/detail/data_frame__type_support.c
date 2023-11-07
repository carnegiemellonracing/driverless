// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cmrdv_interfaces:msg/DataFrame.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cmrdv_interfaces/msg/detail/data_frame__rosidl_typesupport_introspection_c.h"
#include "cmrdv_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cmrdv_interfaces/msg/detail/data_frame__functions.h"
#include "cmrdv_interfaces/msg/detail/data_frame__struct.h"


// Include directives for member types
// Member `zed_left_img`
#include "sensor_msgs/msg/image.h"
// Member `zed_left_img`
#include "sensor_msgs/msg/detail/image__rosidl_typesupport_introspection_c.h"
// Member `zed_pts`
#include "sensor_msgs/msg/point_cloud2.h"
// Member `zed_pts`
#include "sensor_msgs/msg/detail/point_cloud2__rosidl_typesupport_introspection_c.h"
// Member `sbg`
#include "sbg_driver/msg/sbg_gps_pos.h"
// Member `sbg`
#include "sbg_driver/msg/detail/sbg_gps_pos__rosidl_typesupport_introspection_c.h"
// Member `imu`
#include "sbg_driver/msg/sbg_imu_data.h"
// Member `imu`
#include "sbg_driver/msg/detail/sbg_imu_data__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void DataFrame__rosidl_typesupport_introspection_c__DataFrame_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cmrdv_interfaces__msg__DataFrame__init(message_memory);
}

void DataFrame__rosidl_typesupport_introspection_c__DataFrame_fini_function(void * message_memory)
{
  cmrdv_interfaces__msg__DataFrame__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_member_array[4] = {
  {
    "zed_left_img",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__DataFrame, zed_left_img),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "zed_pts",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__DataFrame, zed_pts),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "sbg",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__DataFrame, sbg),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "imu",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__DataFrame, imu),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_members = {
  "cmrdv_interfaces__msg",  // message namespace
  "DataFrame",  // message name
  4,  // number of fields
  sizeof(cmrdv_interfaces__msg__DataFrame),
  DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_member_array,  // message members
  DataFrame__rosidl_typesupport_introspection_c__DataFrame_init_function,  // function to initialize message memory (memory has to be allocated)
  DataFrame__rosidl_typesupport_introspection_c__DataFrame_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_type_support_handle = {
  0,
  &DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cmrdv_interfaces, msg, DataFrame)() {
  DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, Image)();
  DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, PointCloud2)();
  DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sbg_driver, msg, SbgGpsPos)();
  DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sbg_driver, msg, SbgImuData)();
  if (!DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_type_support_handle.typesupport_identifier) {
    DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &DataFrame__rosidl_typesupport_introspection_c__DataFrame_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
