// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/PointArrayStamped.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/point_array_stamped__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/point_array_stamped__functions.h"
#include "eufs_msgs/msg/detail/point_array_stamped__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `points`
#include "geometry_msgs/msg/point.h"
// Member `points`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__PointArrayStamped__init(message_memory);
}

void PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_fini_function(void * message_memory)
{
  eufs_msgs__msg__PointArrayStamped__fini(message_memory);
}

size_t PointArrayStamped__rosidl_typesupport_introspection_c__size_function__Point__points(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * PointArrayStamped__rosidl_typesupport_introspection_c__get_const_function__Point__points(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * PointArrayStamped__rosidl_typesupport_introspection_c__get_function__Point__points(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

bool PointArrayStamped__rosidl_typesupport_introspection_c__resize_function__Point__points(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PointArrayStamped, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "points",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__PointArrayStamped, points),  // bytes offset in struct
    NULL,  // default value
    PointArrayStamped__rosidl_typesupport_introspection_c__size_function__Point__points,  // size() function pointer
    PointArrayStamped__rosidl_typesupport_introspection_c__get_const_function__Point__points,  // get_const(index) function pointer
    PointArrayStamped__rosidl_typesupport_introspection_c__get_function__Point__points,  // get(index) function pointer
    PointArrayStamped__rosidl_typesupport_introspection_c__resize_function__Point__points  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_message_members = {
  "eufs_msgs__msg",  // message namespace
  "PointArrayStamped",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__msg__PointArrayStamped),
  PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_message_member_array,  // message members
  PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_init_function,  // function to initialize message memory (memory has to be allocated)
  PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_message_type_support_handle = {
  0,
  &PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, PointArrayStamped)() {
  PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_message_type_support_handle.typesupport_identifier) {
    PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &PointArrayStamped__rosidl_typesupport_introspection_c__PointArrayStamped_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
