// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:msg/VehicleCommandsStamped.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/msg/detail/vehicle_commands_stamped__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/msg/detail/vehicle_commands_stamped__functions.h"
#include "eufs_msgs/msg/detail/vehicle_commands_stamped__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `commands`
#include "eufs_msgs/msg/vehicle_commands.h"
// Member `commands`
#include "eufs_msgs/msg/detail/vehicle_commands__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__msg__VehicleCommandsStamped__init(message_memory);
}

void VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_fini_function(void * message_memory)
{
  eufs_msgs__msg__VehicleCommandsStamped__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__VehicleCommandsStamped, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "commands",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__msg__VehicleCommandsStamped, commands),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_message_members = {
  "eufs_msgs__msg",  // message namespace
  "VehicleCommandsStamped",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__msg__VehicleCommandsStamped),
  VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_message_member_array,  // message members
  VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_init_function,  // function to initialize message memory (memory has to be allocated)
  VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_message_type_support_handle = {
  0,
  &VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, VehicleCommandsStamped)() {
  VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, VehicleCommands)();
  if (!VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_message_type_support_handle.typesupport_identifier) {
    VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &VehicleCommandsStamped__rosidl_typesupport_introspection_c__VehicleCommandsStamped_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
