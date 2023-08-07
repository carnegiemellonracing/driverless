// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/NodeState.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/node_state__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/node_state__functions.h"
#include "eufs_msgs/msg/detail/node_state__struct.h"


// Include directives for member types
// Member `name`
#include "rosidl_runtime_c/string_functions.h"
// Member `last_heartbeat`
#include "builtin_interfaces/msg/time.h"
// Member `last_heartbeat`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void NodeState__rosidl_typesupport_introspection_c__NodeState_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__NodeState__init(message_memory);
}

void NodeState__rosidl_typesupport_introspection_c__NodeState_fini_function(void * message_memory)
{
  eufs_msgs__msg__NodeState__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember NodeState__rosidl_typesupport_introspection_c__NodeState_message_member_array[6] = {
  {
    "id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__NodeState, id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "name",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__NodeState, name),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "exp_heartbeat",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__NodeState, exp_heartbeat),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "last_heartbeat",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__NodeState, last_heartbeat),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "severity",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__NodeState, severity),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "online",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__NodeState, online),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers NodeState__rosidl_typesupport_introspection_c__NodeState_message_members = {
  "eufs_msgs__msg",  // message namespace
  "NodeState",  // message name
  6,  // number of fields
  sizeof(eufs_msgs__msg__NodeState),
  NodeState__rosidl_typesupport_introspection_c__NodeState_message_member_array,  // message members
  NodeState__rosidl_typesupport_introspection_c__NodeState_init_function,  // function to initialize message memory (memory has to be allocated)
  NodeState__rosidl_typesupport_introspection_c__NodeState_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t NodeState__rosidl_typesupport_introspection_c__NodeState_message_type_support_handle = {
  0,
  &NodeState__rosidl_typesupport_introspection_c__NodeState_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, NodeState)() {
  NodeState__rosidl_typesupport_introspection_c__NodeState_message_member_array[3].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  if (!NodeState__rosidl_typesupport_introspection_c__NodeState_message_type_support_handle.typesupport_identifier) {
    NodeState__rosidl_typesupport_introspection_c__NodeState_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &NodeState__rosidl_typesupport_introspection_c__NodeState_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
