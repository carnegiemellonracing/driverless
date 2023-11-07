// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from cmrdv_interfaces:msg/SimDataFrame.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "cmrdv_interfaces/msg/detail/sim_data_frame__rosidl_typesupport_introspection_c.h"
#include "cmrdv_interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "cmrdv_interfaces/msg/detail/sim_data_frame__functions.h"
#include "cmrdv_interfaces/msg/detail/sim_data_frame__struct.h"


// Include directives for member types
// Member `gt_cones`
#include "eufs_msgs/msg/cone_array_with_covariance.h"
// Member `gt_cones`
#include "eufs_msgs/msg/detail/cone_array_with_covariance__rosidl_typesupport_introspection_c.h"
// Member `zed_left_img`
#include "sensor_msgs/msg/image.h"
// Member `zed_left_img`
#include "sensor_msgs/msg/detail/image__rosidl_typesupport_introspection_c.h"
// Member `vlp16_pts`
// Member `zed_pts`
#include "sensor_msgs/msg/point_cloud2.h"
// Member `vlp16_pts`
// Member `zed_pts`
#include "sensor_msgs/msg/detail/point_cloud2__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  cmrdv_interfaces__msg__SimDataFrame__init(message_memory);
}

void SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_fini_function(void * message_memory)
{
  cmrdv_interfaces__msg__SimDataFrame__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_member_array[4] = {
  {
    "gt_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__SimDataFrame, gt_cones),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "zed_left_img",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__SimDataFrame, zed_left_img),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vlp16_pts",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces__msg__SimDataFrame, vlp16_pts),  // bytes offset in struct
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
    offsetof(cmrdv_interfaces__msg__SimDataFrame, zed_pts),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_members = {
  "cmrdv_interfaces__msg",  // message namespace
  "SimDataFrame",  // message name
  4,  // number of fields
  sizeof(cmrdv_interfaces__msg__SimDataFrame),
  SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_member_array,  // message members
  SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_init_function,  // function to initialize message memory (memory has to be allocated)
  SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_type_support_handle = {
  0,
  &SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, cmrdv_interfaces, msg, SimDataFrame)() {
  SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, ConeArrayWithCovariance)();
  SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, Image)();
  SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, PointCloud2)();
  SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, PointCloud2)();
  if (!SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_type_support_handle.typesupport_identifier) {
    SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &SimDataFrame__rosidl_typesupport_introspection_c__SimDataFrame_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
