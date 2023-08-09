// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/ConeArrayWithCovariance.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/cone_array_with_covariance__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/cone_array_with_covariance__functions.h"
#include "eufs_msgs/msg/detail/cone_array_with_covariance__struct.h"


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
#include "eufs_msgs/msg/cone_with_covariance.h"
// Member `blue_cones`
// Member `yellow_cones`
// Member `orange_cones`
// Member `big_orange_cones`
// Member `unknown_color_cones`
#include "eufs_msgs/msg/detail/cone_with_covariance__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__ConeArrayWithCovariance__init(message_memory);
}

void ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_fini_function(void * message_memory)
{
  eufs_msgs__msg__ConeArrayWithCovariance__fini(message_memory);
}

size_t ConeArrayWithCovariance__rosidl_typesupport_introspection_c__size_function__ConeWithCovariance__blue_cones(
  const void * untyped_member)
{
  const eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (const eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return member->size;
}

const void * ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_const_function__ConeWithCovariance__blue_cones(
  const void * untyped_member, size_t index)
{
  const eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (const eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_function__ConeWithCovariance__blue_cones(
  void * untyped_member, size_t index)
{
  eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ConeArrayWithCovariance__rosidl_typesupport_introspection_c__resize_function__ConeWithCovariance__blue_cones(
  void * untyped_member, size_t size)
{
  eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  eufs_msgs__msg__ConeWithCovariance__Sequence__fini(member);
  return eufs_msgs__msg__ConeWithCovariance__Sequence__init(member, size);
}

size_t ConeArrayWithCovariance__rosidl_typesupport_introspection_c__size_function__ConeWithCovariance__yellow_cones(
  const void * untyped_member)
{
  const eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (const eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return member->size;
}

const void * ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_const_function__ConeWithCovariance__yellow_cones(
  const void * untyped_member, size_t index)
{
  const eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (const eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_function__ConeWithCovariance__yellow_cones(
  void * untyped_member, size_t index)
{
  eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ConeArrayWithCovariance__rosidl_typesupport_introspection_c__resize_function__ConeWithCovariance__yellow_cones(
  void * untyped_member, size_t size)
{
  eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  eufs_msgs__msg__ConeWithCovariance__Sequence__fini(member);
  return eufs_msgs__msg__ConeWithCovariance__Sequence__init(member, size);
}

size_t ConeArrayWithCovariance__rosidl_typesupport_introspection_c__size_function__ConeWithCovariance__orange_cones(
  const void * untyped_member)
{
  const eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (const eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return member->size;
}

const void * ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_const_function__ConeWithCovariance__orange_cones(
  const void * untyped_member, size_t index)
{
  const eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (const eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_function__ConeWithCovariance__orange_cones(
  void * untyped_member, size_t index)
{
  eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ConeArrayWithCovariance__rosidl_typesupport_introspection_c__resize_function__ConeWithCovariance__orange_cones(
  void * untyped_member, size_t size)
{
  eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  eufs_msgs__msg__ConeWithCovariance__Sequence__fini(member);
  return eufs_msgs__msg__ConeWithCovariance__Sequence__init(member, size);
}

size_t ConeArrayWithCovariance__rosidl_typesupport_introspection_c__size_function__ConeWithCovariance__big_orange_cones(
  const void * untyped_member)
{
  const eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (const eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return member->size;
}

const void * ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_const_function__ConeWithCovariance__big_orange_cones(
  const void * untyped_member, size_t index)
{
  const eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (const eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_function__ConeWithCovariance__big_orange_cones(
  void * untyped_member, size_t index)
{
  eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ConeArrayWithCovariance__rosidl_typesupport_introspection_c__resize_function__ConeWithCovariance__big_orange_cones(
  void * untyped_member, size_t size)
{
  eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  eufs_msgs__msg__ConeWithCovariance__Sequence__fini(member);
  return eufs_msgs__msg__ConeWithCovariance__Sequence__init(member, size);
}

size_t ConeArrayWithCovariance__rosidl_typesupport_introspection_c__size_function__ConeWithCovariance__unknown_color_cones(
  const void * untyped_member)
{
  const eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (const eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return member->size;
}

const void * ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_const_function__ConeWithCovariance__unknown_color_cones(
  const void * untyped_member, size_t index)
{
  const eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (const eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_function__ConeWithCovariance__unknown_color_cones(
  void * untyped_member, size_t index)
{
  eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ConeArrayWithCovariance__rosidl_typesupport_introspection_c__resize_function__ConeWithCovariance__unknown_color_cones(
  void * untyped_member, size_t size)
{
  eufs_msgs__msg__ConeWithCovariance__Sequence * member =
    (eufs_msgs__msg__ConeWithCovariance__Sequence *)(untyped_member);
  eufs_msgs__msg__ConeWithCovariance__Sequence__fini(member);
  return eufs_msgs__msg__ConeWithCovariance__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_member_array[6] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__ConeArrayWithCovariance, header),  // bytes offset in struct
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
    offsetof(eufs_msgs__msg__ConeArrayWithCovariance, blue_cones),  // bytes offset in struct
    NULL,  // default value
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__size_function__ConeWithCovariance__blue_cones,  // size() function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_const_function__ConeWithCovariance__blue_cones,  // get_const(index) function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_function__ConeWithCovariance__blue_cones,  // get(index) function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__resize_function__ConeWithCovariance__blue_cones  // resize(index) function pointer
  },
  {
    "yellow_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__ConeArrayWithCovariance, yellow_cones),  // bytes offset in struct
    NULL,  // default value
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__size_function__ConeWithCovariance__yellow_cones,  // size() function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_const_function__ConeWithCovariance__yellow_cones,  // get_const(index) function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_function__ConeWithCovariance__yellow_cones,  // get(index) function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__resize_function__ConeWithCovariance__yellow_cones  // resize(index) function pointer
  },
  {
    "orange_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__ConeArrayWithCovariance, orange_cones),  // bytes offset in struct
    NULL,  // default value
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__size_function__ConeWithCovariance__orange_cones,  // size() function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_const_function__ConeWithCovariance__orange_cones,  // get_const(index) function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_function__ConeWithCovariance__orange_cones,  // get(index) function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__resize_function__ConeWithCovariance__orange_cones  // resize(index) function pointer
  },
  {
    "big_orange_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__ConeArrayWithCovariance, big_orange_cones),  // bytes offset in struct
    NULL,  // default value
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__size_function__ConeWithCovariance__big_orange_cones,  // size() function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_const_function__ConeWithCovariance__big_orange_cones,  // get_const(index) function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_function__ConeWithCovariance__big_orange_cones,  // get(index) function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__resize_function__ConeWithCovariance__big_orange_cones  // resize(index) function pointer
  },
  {
    "unknown_color_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__ConeArrayWithCovariance, unknown_color_cones),  // bytes offset in struct
    NULL,  // default value
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__size_function__ConeWithCovariance__unknown_color_cones,  // size() function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_const_function__ConeWithCovariance__unknown_color_cones,  // get_const(index) function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__get_function__ConeWithCovariance__unknown_color_cones,  // get(index) function pointer
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__resize_function__ConeWithCovariance__unknown_color_cones  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_members = {
  "eufs_msgs__msg",  // message namespace
  "ConeArrayWithCovariance",  // message name
  6,  // number of fields
  sizeof(eufs_msgs__msg__ConeArrayWithCovariance),
  ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_member_array,  // message members
  ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_init_function,  // function to initialize message memory (memory has to be allocated)
  ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_type_support_handle = {
  0,
  &ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, ConeArrayWithCovariance)() {
  ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, ConeWithCovariance)();
  ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, ConeWithCovariance)();
  ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, ConeWithCovariance)();
  ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_member_array[4].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, ConeWithCovariance)();
  ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_member_array[5].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, ConeWithCovariance)();
  if (!ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_type_support_handle.typesupport_identifier) {
    ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ConeArrayWithCovariance__rosidl_typesupport_introspection_c__ConeArrayWithCovariance_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
