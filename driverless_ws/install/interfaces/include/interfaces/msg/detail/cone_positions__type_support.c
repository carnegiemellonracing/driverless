// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from interfaces:msg/ConePositions.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "interfaces/msg/detail/cone_positions__rosidl_typesupport_introspection_c.h"
#include "interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "interfaces/msg/detail/cone_positions__functions.h"
#include "interfaces/msg/detail/cone_positions__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `cone_positions`
#include "std_msgs/msg/float32.h"
// Member `cone_positions`
#include "std_msgs/msg/detail/float32__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void ConePositions__rosidl_typesupport_introspection_c__ConePositions_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  interfaces__msg__ConePositions__init(message_memory);
}

void ConePositions__rosidl_typesupport_introspection_c__ConePositions_fini_function(void * message_memory)
{
  interfaces__msg__ConePositions__fini(message_memory);
}

size_t ConePositions__rosidl_typesupport_introspection_c__size_function__Float32__cone_positions(
  const void * untyped_member)
{
  const std_msgs__msg__Float32__Sequence * member =
    (const std_msgs__msg__Float32__Sequence *)(untyped_member);
  return member->size;
}

const void * ConePositions__rosidl_typesupport_introspection_c__get_const_function__Float32__cone_positions(
  const void * untyped_member, size_t index)
{
  const std_msgs__msg__Float32__Sequence * member =
    (const std_msgs__msg__Float32__Sequence *)(untyped_member);
  return &member->data[index];
}

void * ConePositions__rosidl_typesupport_introspection_c__get_function__Float32__cone_positions(
  void * untyped_member, size_t index)
{
  std_msgs__msg__Float32__Sequence * member =
    (std_msgs__msg__Float32__Sequence *)(untyped_member);
  return &member->data[index];
}

bool ConePositions__rosidl_typesupport_introspection_c__resize_function__Float32__cone_positions(
  void * untyped_member, size_t size)
{
  std_msgs__msg__Float32__Sequence * member =
    (std_msgs__msg__Float32__Sequence *)(untyped_member);
  std_msgs__msg__Float32__Sequence__fini(member);
  return std_msgs__msg__Float32__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember ConePositions__rosidl_typesupport_introspection_c__ConePositions_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(interfaces__msg__ConePositions, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "cone_positions",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(interfaces__msg__ConePositions, cone_positions),  // bytes offset in struct
    NULL,  // default value
    ConePositions__rosidl_typesupport_introspection_c__size_function__Float32__cone_positions,  // size() function pointer
    ConePositions__rosidl_typesupport_introspection_c__get_const_function__Float32__cone_positions,  // get_const(index) function pointer
    ConePositions__rosidl_typesupport_introspection_c__get_function__Float32__cone_positions,  // get(index) function pointer
    ConePositions__rosidl_typesupport_introspection_c__resize_function__Float32__cone_positions  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ConePositions__rosidl_typesupport_introspection_c__ConePositions_message_members = {
  "interfaces__msg",  // message namespace
  "ConePositions",  // message name
  2,  // number of fields
  sizeof(interfaces__msg__ConePositions),
  ConePositions__rosidl_typesupport_introspection_c__ConePositions_message_member_array,  // message members
  ConePositions__rosidl_typesupport_introspection_c__ConePositions_init_function,  // function to initialize message memory (memory has to be allocated)
  ConePositions__rosidl_typesupport_introspection_c__ConePositions_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ConePositions__rosidl_typesupport_introspection_c__ConePositions_message_type_support_handle = {
  0,
  &ConePositions__rosidl_typesupport_introspection_c__ConePositions_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, interfaces, msg, ConePositions)() {
  ConePositions__rosidl_typesupport_introspection_c__ConePositions_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  ConePositions__rosidl_typesupport_introspection_c__ConePositions_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Float32)();
  if (!ConePositions__rosidl_typesupport_introspection_c__ConePositions_message_type_support_handle.typesupport_identifier) {
    ConePositions__rosidl_typesupport_introspection_c__ConePositions_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ConePositions__rosidl_typesupport_introspection_c__ConePositions_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
