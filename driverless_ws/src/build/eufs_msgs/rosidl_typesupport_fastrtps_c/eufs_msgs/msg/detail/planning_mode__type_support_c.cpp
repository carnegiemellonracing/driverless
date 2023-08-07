// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/PlanningMode.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/planning_mode__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/planning_mode__struct.h"
#include "eufs_msgs/msg/detail/planning_mode__functions.h"
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


using _PlanningMode__ros_msg_type = eufs_msgs__msg__PlanningMode;

static bool _PlanningMode__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _PlanningMode__ros_msg_type * ros_message = static_cast<const _PlanningMode__ros_msg_type *>(untyped_ros_message);
  // Field name: mode
  {
    cdr << ros_message->mode;
  }

  return true;
}

static bool _PlanningMode__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _PlanningMode__ros_msg_type * ros_message = static_cast<_PlanningMode__ros_msg_type *>(untyped_ros_message);
  // Field name: mode
  {
    cdr >> ros_message->mode;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__PlanningMode(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PlanningMode__ros_msg_type * ros_message = static_cast<const _PlanningMode__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name mode
  {
    size_t item_size = sizeof(ros_message->mode);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _PlanningMode__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__PlanningMode(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__PlanningMode(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: mode
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  return current_alignment - initial_alignment;
}

static size_t _PlanningMode__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__PlanningMode(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_PlanningMode = {
  "eufs_msgs::msg",
  "PlanningMode",
  _PlanningMode__cdr_serialize,
  _PlanningMode__cdr_deserialize,
  _PlanningMode__get_serialized_size,
  _PlanningMode__max_serialized_size
};

static rosidl_message_type_support_t _PlanningMode__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PlanningMode,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, PlanningMode)() {
  return &_PlanningMode__type_support;
}

#if defined(__cplusplus)
}
#endif
