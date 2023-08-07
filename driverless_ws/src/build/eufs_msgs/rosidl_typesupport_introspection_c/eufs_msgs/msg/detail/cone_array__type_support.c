// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/ConeArray.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/cone_array__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/cone_array__functions.h"
#include "eufs_msgs/msg/detail/cone_array__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `blue_cones`
// Member `yellow_cones`
// Member `orange_cones`
// Member `big_orange_cones`
// Member `unknown_color_cones`
#include "geometry_msgs/msg/point.h"
// Member `blue_cones`
// Member `yellow_cones`
// Member `orange_cones`
// Member `big_orange_cones`
// Member `unknown_color_cones`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ConeArray__rosidl_typesupport_introspection_c__ConeArray_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__ConeArray__init(message_memory);
}

void ConeArray__rosidl_typesupport_introspection_c__ConeArray_fini_function(void * message_memory)
{
  eufs_msgs__msg__ConeArray__fini(message_memory);
}

size_t ConeArray__rosidl_typesupport_introspection_c__size_function__Point__blue_cones(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * ConeArray__rosidl_typesupport_introspection_c__get_const_function__Point__blue_cones(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ConeArray__rosidl_typesupport_introspection_c__get_function__Point__blue_cones(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ConeArray__rosidl_typesupport_introspection_c__resize_function__Point__blue_cones(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

size_t ConeArray__rosidl_typesupport_introspection_c__size_function__Point__yellow_cones(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * ConeArray__rosidl_typesupport_introspection_c__get_const_function__Point__yellow_cones(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ConeArray__rosidl_typesupport_introspection_c__get_function__Point__yellow_cones(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ConeArray__rosidl_typesupport_introspection_c__resize_function__Point__yellow_cones(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

size_t ConeArray__rosidl_typesupport_introspection_c__size_function__Point__orange_cones(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * ConeArray__rosidl_typesupport_introspection_c__get_const_function__Point__orange_cones(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ConeArray__rosidl_typesupport_introspection_c__get_function__Point__orange_cones(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ConeArray__rosidl_typesupport_introspection_c__resize_function__Point__orange_cones(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

size_t ConeArray__rosidl_typesupport_introspection_c__size_function__Point__big_orange_cones(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * ConeArray__rosidl_typesupport_introspection_c__get_const_function__Point__big_orange_cones(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ConeArray__rosidl_typesupport_introspection_c__get_function__Point__big_orange_cones(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ConeArray__rosidl_typesupport_introspection_c__resize_function__Point__big_orange_cones(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

size_t ConeArray__rosidl_typesupport_introspection_c__size_function__Point__unknown_color_cones(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * ConeArray__rosidl_typesupport_introspection_c__get_const_function__Point__unknown_color_cones(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ConeArray__rosidl_typesupport_introspection_c__get_function__Point__unknown_color_cones(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ConeArray__rosidl_typesupport_introspection_c__resize_function__Point__unknown_color_cones(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_member_array[6] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__ConeArray, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "blue_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__ConeArray, blue_cones),  // bytes offset in struct
    NULL,  // default value
    ConeArray__rosidl_typesupport_introspection_c__size_function__Point__blue_cones,  // size() function pointer
    ConeArray__rosidl_typesupport_introspection_c__get_const_function__Point__blue_cones,  // get_const(index) function pointer
    ConeArray__rosidl_typesupport_introspection_c__get_function__Point__blue_cones,  // get(index) function pointer
    ConeArray__rosidl_typesupport_introspection_c__resize_function__Point__blue_cones  // resize(index) function pointer
  },
  {
    "yellow_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__ConeArray, yellow_cones),  // bytes offset in struct
    NULL,  // default value
    ConeArray__rosidl_typesupport_introspection_c__size_function__Point__yellow_cones,  // size() function pointer
    ConeArray__rosidl_typesupport_introspection_c__get_const_function__Point__yellow_cones,  // get_const(index) function pointer
    ConeArray__rosidl_typesupport_introspection_c__get_function__Point__yellow_cones,  // get(index) function pointer
    ConeArray__rosidl_typesupport_introspection_c__resize_function__Point__yellow_cones  // resize(index) function pointer
  },
  {
    "orange_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__ConeArray, orange_cones),  // bytes offset in struct
    NULL,  // default value
    ConeArray__rosidl_typesupport_introspection_c__size_function__Point__orange_cones,  // size() function pointer
    ConeArray__rosidl_typesupport_introspection_c__get_const_function__Point__orange_cones,  // get_const(index) function pointer
    ConeArray__rosidl_typesupport_introspection_c__get_function__Point__orange_cones,  // get(index) function pointer
    ConeArray__rosidl_typesupport_introspection_c__resize_function__Point__orange_cones  // resize(index) function pointer
  },
  {
    "big_orange_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__ConeArray, big_orange_cones),  // bytes offset in struct
    NULL,  // default value
    ConeArray__rosidl_typesupport_introspection_c__size_function__Point__big_orange_cones,  // size() function pointer
    ConeArray__rosidl_typesupport_introspection_c__get_const_function__Point__big_orange_cones,  // get_const(index) function pointer
    ConeArray__rosidl_typesupport_introspection_c__get_function__Point__big_orange_cones,  // get(index) function pointer
    ConeArray__rosidl_typesupport_introspection_c__resize_function__Point__big_orange_cones  // resize(index) function pointer
  },
  {
    "unknown_color_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__ConeArray, unknown_color_cones),  // bytes offset in struct
    NULL,  // default value
    ConeArray__rosidl_typesupport_introspection_c__size_function__Point__unknown_color_cones,  // size() function pointer
    ConeArray__rosidl_typesupport_introspection_c__get_const_function__Point__unknown_color_cones,  // get_const(index) function pointer
    ConeArray__rosidl_typesupport_introspection_c__get_function__Point__unknown_color_cones,  // get(index) function pointer
    ConeArray__rosidl_typesupport_introspection_c__resize_function__Point__unknown_color_cones  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_members = {
  "eufs_msgs__msg",  // message namespace
  "ConeArray",  // message name
  6,  // number of fields
  sizeof(eufs_msgs__msg__ConeArray),
  ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_member_array,  // message members
  ConeArray__rosidl_typesupport_introspection_c__ConeArray_init_function,  // function to initialize message memory (memory has to be allocated)
  ConeArray__rosidl_typesupport_introspection_c__ConeArray_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_type_support_handle = {
  0,
  &ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, ConeArray)() {
  ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_member_array[4].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_member_array[5].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_type_support_handle.typesupport_identifier) {
    ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ConeArray__rosidl_typesupport_introspection_c__ConeArray_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
