// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/StateMachineState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/state_machine_state__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/state_machine_state__struct.h"
#include "eufs_msgs/msg/detail/state_machine_state__functions.h"
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

#include "rosidl_runtime_c/string.h"  // state_str
#include "rosidl_runtime_c/string_functions.h"  // state_str

// forward declare type support functions


using _StateMachineState__ros_msg_type = eufs_msgs__msg__StateMachineState;

static bool _StateMachineState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _StateMachineState__ros_msg_type * ros_message = static_cast<const _StateMachineState__ros_msg_type *>(untyped_ros_message);
  // Field name: state
  {
    cdr << ros_message->state;
  }

  // Field name: state_str
  {
    const rosidl_runtime_c__String * str = &ros_message->state_str;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  return true;
}

static bool _StateMachineState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _StateMachineState__ros_msg_type * ros_message = static_cast<_StateMachineState__ros_msg_type *>(untyped_ros_message);
  // Field name: state
  {
    cdr >> ros_message->state;
  }

  // Field name: state_str
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->state_str.data) {
      rosidl_runtime_c__String__init(&ros_message->state_str);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->state_str,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'state_str'\n");
      return false;
    }
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__StateMachineState(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _StateMachineState__ros_msg_type * ros_message = static_cast<const _StateMachineState__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name state
  {
    size_t item_size = sizeof(ros_message->state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name state_str
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->state_str.size + 1);

  return current_alignment - initial_alignment;
}

static uint32_t _StateMachineState__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__StateMachineState(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__StateMachineState(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: state
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }
  // member: state_str
  {
    size_t array_size = 1;

    full_bounded = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  return current_alignment - initial_alignment;
}

static size_t _StateMachineState__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__StateMachineState(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_StateMachineState = {
  "eufs_msgs::msg",
  "StateMachineState",
  _StateMachineState__cdr_serialize,
  _StateMachineState__cdr_deserialize,
  _StateMachineState__get_serialized_size,
  _StateMachineState__max_serialized_size
};

static rosidl_message_type_support_t _StateMachineState__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_StateMachineState,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, StateMachineState)() {
  return &_StateMachineState__type_support;
}

#if defined(__cplusplus)
}
#endif
