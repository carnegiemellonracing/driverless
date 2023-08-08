// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/SystemStatus.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/system_status__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/system_status__functions.h"
#include "eufs_msgs/msg/detail/system_status__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `topic_statuses`
#include "eufs_msgs/msg/topic_status.h"
// Member `topic_statuses`
#include "eufs_msgs/msg/detail/topic_status__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__SystemStatus__init(message_memory);
}

void SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_fini_function(void * message_memory)
{
  eufs_msgs__msg__SystemStatus__fini(message_memory);
}

size_t SystemStatus__rosidl_typesupport_introspection_c__size_function__TopicStatus__topic_statuses(
  const void * untyped_member)
{
  const eufs_msgs__msg__TopicStatus__Sequence * member =
    (const eufs_msgs__msg__TopicStatus__Sequence *)(untyped_member);
  return member->size;
}

const void * SystemStatus__rosidl_typesupport_introspection_c__get_const_function__TopicStatus__topic_statuses(
  const void * untyped_member, size_t index)
{
  const eufs_msgs__msg__TopicStatus__Sequence * member =
    (const eufs_msgs__msg__TopicStatus__Sequence *)(untyped_member);
  return &member->data[index];
}

void * SystemStatus__rosidl_typesupport_introspection_c__get_function__TopicStatus__topic_statuses(
  void * untyped_member, size_t index)
{
  eufs_msgs__msg__TopicStatus__Sequence * member =
    (eufs_msgs__msg__TopicStatus__Sequence *)(untyped_member);
  return &member->data[index];
}

bool SystemStatus__rosidl_typesupport_introspection_c__resize_function__TopicStatus__topic_statuses(
  void * untyped_member, size_t size)
{
  eufs_msgs__msg__TopicStatus__Sequence * member =
    (eufs_msgs__msg__TopicStatus__Sequence *)(untyped_member);
  eufs_msgs__msg__TopicStatus__Sequence__fini(member);
  return eufs_msgs__msg__TopicStatus__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__SystemStatus, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "topic_statuses",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__SystemStatus, topic_statuses),  // bytes offset in struct
    NULL,  // default value
    SystemStatus__rosidl_typesupport_introspection_c__size_function__TopicStatus__topic_statuses,  // size() function pointer
    SystemStatus__rosidl_typesupport_introspection_c__get_const_function__TopicStatus__topic_statuses,  // get_const(index) function pointer
    SystemStatus__rosidl_typesupport_introspection_c__get_function__TopicStatus__topic_statuses,  // get(index) function pointer
    SystemStatus__rosidl_typesupport_introspection_c__resize_function__TopicStatus__topic_statuses  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_message_members = {
  "eufs_msgs__msg",  // message namespace
  "SystemStatus",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__msg__SystemStatus),
  SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_message_member_array,  // message members
  SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_init_function,  // function to initialize message memory (memory has to be allocated)
  SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_message_type_support_handle = {
  0,
  &SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, SystemStatus)() {
  SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, TopicStatus)();
  if (!SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_message_type_support_handle.typesupport_identifier) {
    SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &SystemStatus__rosidl_typesupport_introspection_c__SystemStatus_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
