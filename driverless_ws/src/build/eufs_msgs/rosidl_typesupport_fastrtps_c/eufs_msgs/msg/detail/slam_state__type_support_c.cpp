// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/SLAMState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/slam_state__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/slam_state__struct.h"
#include "eufs_msgs/msg/detail/slam_state__functions.h"
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

#include "rosidl_runtime_c/string.h"  // status
#include "rosidl_runtime_c/string_functions.h"  // status

// forward declare type support functions


using _SLAMState__ros_msg_type = eufs_msgs__msg__SLAMState;

static bool _SLAMState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _SLAMState__ros_msg_type * ros_message = static_cast<const _SLAMState__ros_msg_type *>(untyped_ros_message);
  // Field name: loop_closed
  {
    cdr << (ros_message->loop_closed ? true : false);
  }

  // Field name: laps
  {
    cdr << ros_message->laps;
  }

  // Field name: status
  {
    const rosidl_runtime_c__String * str = &ros_message->status;
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

  // Field name: state
  {
    cdr << ros_message->state;
  }

  return true;
}

static bool _SLAMState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _SLAMState__ros_msg_type * ros_message = static_cast<_SLAMState__ros_msg_type *>(untyped_ros_message);
  // Field name: loop_closed
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->loop_closed = tmp ? true : false;
  }

  // Field name: laps
  {
    cdr >> ros_message->laps;
  }

  // Field name: status
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->status.data) {
      rosidl_runtime_c__String__init(&ros_message->status);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->status,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'status'\n");
      return false;
    }
  }

  // Field name: state
  {
    cdr >> ros_message->state;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__SLAMState(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _SLAMState__ros_msg_type * ros_message = static_cast<const _SLAMState__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name loop_closed
  {
    size_t item_size = sizeof(ros_message->loop_closed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name laps
  {
    size_t item_size = sizeof(ros_message->laps);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name status
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->status.size + 1);
  // field.name state
  {
    size_t item_size = sizeof(ros_message->state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _SLAMState__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__SLAMState(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__SLAMState(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: loop_closed
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: laps
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }
  // member: status
  {
    size_t array_size = 1;

    full_bounded = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: state
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  return current_alignment - initial_alignment;
}

static size_t _SLAMState__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__SLAMState(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_SLAMState = {
  "eufs_msgs::msg",
  "SLAMState",
  _SLAMState__cdr_serialize,
  _SLAMState__cdr_deserialize,
  _SLAMState__get_serialized_size,
  _SLAMState__max_serialized_size
};

static rosidl_message_type_support_t _SLAMState__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_SLAMState,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, SLAMState)() {
  return &_SLAMState__type_support;
}

#if defined(__cplusplus)
}
#endif
