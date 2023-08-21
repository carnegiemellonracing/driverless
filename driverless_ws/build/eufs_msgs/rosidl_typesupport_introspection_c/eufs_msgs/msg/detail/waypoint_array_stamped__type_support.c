// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/WaypointArrayStamped.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/waypoint_array_stamped__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/waypoint_array_stamped__functions.h"
#include "eufs_msgs/msg/detail/waypoint_array_stamped__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `waypoints`
#include "eufs_msgs/msg/waypoint.h"
// Member `waypoints`
#include "eufs_msgs/msg/detail/waypoint__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__WaypointArrayStamped__init(message_memory);
}

void WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_fini_function(void * message_memory)
{
  eufs_msgs__msg__WaypointArrayStamped__fini(message_memory);
}

size_t WaypointArrayStamped__rosidl_typesupport_introspection_c__size_function__Waypoint__waypoints(
  const void * untyped_member)
{
  const eufs_msgs__msg__Waypoint__Sequence * member =
    (const eufs_msgs__msg__Waypoint__Sequence *)(untyped_member);
  return member->size;
}

const void * WaypointArrayStamped__rosidl_typesupport_introspection_c__get_const_function__Waypoint__waypoints(
  const void * untyped_member, size_t index)
{
  const eufs_msgs__msg__Waypoint__Sequence * member =
    (const eufs_msgs__msg__Waypoint__Sequence *)(untyped_member);
  return &member->data[index];
}

void * WaypointArrayStamped__rosidl_typesupport_introspection_c__get_function__Waypoint__waypoints(
  void * untyped_member, size_t index)
{
  eufs_msgs__msg__Waypoint__Sequence * member =
    (eufs_msgs__msg__Waypoint__Sequence *)(untyped_member);
  return &member->data[index];
}

bool WaypointArrayStamped__rosidl_typesupport_introspection_c__resize_function__Waypoint__waypoints(
  void * untyped_member, size_t size)
{
  eufs_msgs__msg__Waypoint__Sequence * member =
    (eufs_msgs__msg__Waypoint__Sequence *)(untyped_member);
  eufs_msgs__msg__Waypoint__Sequence__fini(member);
  return eufs_msgs__msg__Waypoint__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WaypointArrayStamped, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "waypoints",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__WaypointArrayStamped, waypoints),  // bytes offset in struct
    NULL,  // default value
    WaypointArrayStamped__rosidl_typesupport_introspection_c__size_function__Waypoint__waypoints,  // size() function pointer
    WaypointArrayStamped__rosidl_typesupport_introspection_c__get_const_function__Waypoint__waypoints,  // get_const(index) function pointer
    WaypointArrayStamped__rosidl_typesupport_introspection_c__get_function__Waypoint__waypoints,  // get(index) function pointer
    WaypointArrayStamped__rosidl_typesupport_introspection_c__resize_function__Waypoint__waypoints  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_message_members = {
  "eufs_msgs__msg",  // message namespace
  "WaypointArrayStamped",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__msg__WaypointArrayStamped),
  WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_message_member_array,  // message members
  WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_init_function,  // function to initialize message memory (memory has to be allocated)
  WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_message_type_support_handle = {
  0,
  &WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, WaypointArrayStamped)() {
  WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, Waypoint)();
  if (!WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_message_type_support_handle.typesupport_identifier) {
    WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &WaypointArrayStamped__rosidl_typesupport_introspection_c__WaypointArrayStamped_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
