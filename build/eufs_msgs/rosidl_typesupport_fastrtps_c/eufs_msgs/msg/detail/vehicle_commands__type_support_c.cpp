// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/VehicleCommands.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/vehicle_commands__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/vehicle_commands__struct.h"
#include "eufs_msgs/msg/detail/vehicle_commands__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _VehicleCommands__ros_msg_type = eufs_msgs__msg__VehicleCommands;

static bool _VehicleCommands__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _VehicleCommands__ros_msg_type * ros_message = static_cast<const _VehicleCommands__ros_msg_type *>(untyped_ros_message);
  // Field name: handshake
  {
    cdr << ros_message->handshake;
  }

  // Field name: ebs
  {
    cdr << ros_message->ebs;
  }

  // Field name: direction
  {
    cdr << ros_message->direction;
  }

  // Field name: mission_status
  {
    cdr << ros_message->mission_status;
  }

  // Field name: braking
  {
    cdr << ros_message->braking;
  }

  // Field name: torque
  {
    cdr << ros_message->torque;
  }

  // Field name: steering
  {
    cdr << ros_message->steering;
  }

  // Field name: rpm
  {
    cdr << ros_message->rpm;
  }

  return true;
}

static bool _VehicleCommands__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _VehicleCommands__ros_msg_type * ros_message = static_cast<_VehicleCommands__ros_msg_type *>(untyped_ros_message);
  // Field name: handshake
  {
    cdr >> ros_message->handshake;
  }

  // Field name: ebs
  {
    cdr >> ros_message->ebs;
  }

  // Field name: direction
  {
    cdr >> ros_message->direction;
  }

  // Field name: mission_status
  {
    cdr >> ros_message->mission_status;
  }

  // Field name: braking
  {
    cdr >> ros_message->braking;
  }

  // Field name: torque
  {
    cdr >> ros_message->torque;
  }

  // Field name: steering
  {
    cdr >> ros_message->steering;
  }

  // Field name: rpm
  {
    cdr >> ros_message->rpm;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__VehicleCommands(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _VehicleCommands__ros_msg_type * ros_message = static_cast<const _VehicleCommands__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name handshake
  {
    size_t item_size = sizeof(ros_message->handshake);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ebs
  {
    size_t item_size = sizeof(ros_message->ebs);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name direction
  {
    size_t item_size = sizeof(ros_message->direction);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name mission_status
  {
    size_t item_size = sizeof(ros_message->mission_status);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name braking
  {
    size_t item_size = sizeof(ros_message->braking);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name torque
  {
    size_t item_size = sizeof(ros_message->torque);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name steering
  {
    size_t item_size = sizeof(ros_message->steering);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name rpm
  {
    size_t item_size = sizeof(ros_message->rpm);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _VehicleCommands__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__VehicleCommands(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__VehicleCommands(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: handshake
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: ebs
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: direction
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: mission_status
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: braking
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: torque
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: steering
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: rpm
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static size_t _VehicleCommands__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__VehicleCommands(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_VehicleCommands = {
  "eufs_msgs::msg",
  "VehicleCommands",
  _VehicleCommands__cdr_serialize,
  _VehicleCommands__cdr_deserialize,
  _VehicleCommands__get_serialized_size,
  _VehicleCommands__max_serialized_size
};

static rosidl_message_type_support_t _VehicleCommands__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_VehicleCommands,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, VehicleCommands)() {
  return &_VehicleCommands__type_support;
}

#if defined(__cplusplus)
}
#endif
