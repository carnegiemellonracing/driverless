// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/PathIntegralStats.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/path_integral_stats__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/path_integral_stats__struct.h"
#include "eufs_msgs/msg/detail/path_integral_stats__functions.h"
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

#include "eufs_msgs/msg/detail/lap_stats__functions.h"  // stats
#include "eufs_msgs/msg/detail/path_integral_params__functions.h"  // params
#include "rosidl_runtime_c/string.h"  // tag
#include "rosidl_runtime_c/string_functions.h"  // tag
#include "std_msgs/msg/detail/header__functions.h"  // header

// forward declare type support functions
size_t get_serialized_size_eufs_msgs__msg__LapStats(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_eufs_msgs__msg__LapStats(
  bool & full_bounded,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, LapStats)();
size_t get_serialized_size_eufs_msgs__msg__PathIntegralParams(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_eufs_msgs__msg__PathIntegralParams(
  bool & full_bounded,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, PathIntegralParams)();
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_eufs_msgs
size_t get_serialized_size_std_msgs__msg__Header(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_eufs_msgs
size_t max_serialized_size_std_msgs__msg__Header(
  bool & full_bounded,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_eufs_msgs
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, std_msgs, msg, Header)();


using _PathIntegralStats__ros_msg_type = eufs_msgs__msg__PathIntegralStats;

static bool _PathIntegralStats__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _PathIntegralStats__ros_msg_type * ros_message = static_cast<const _PathIntegralStats__ros_msg_type *>(untyped_ros_message);
  // Field name: header
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Header
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->header, cdr))
    {
      return false;
    }
  }

  // Field name: tag
  {
    const rosidl_runtime_c__String * str = &ros_message->tag;
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

  // Field name: params
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, PathIntegralParams
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->params, cdr))
    {
      return false;
    }
  }

  // Field name: stats
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, LapStats
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->stats, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _PathIntegralStats__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _PathIntegralStats__ros_msg_type * ros_message = static_cast<_PathIntegralStats__ros_msg_type *>(untyped_ros_message);
  // Field name: header
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, std_msgs, msg, Header
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->header))
    {
      return false;
    }
  }

  // Field name: tag
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->tag.data) {
      rosidl_runtime_c__String__init(&ros_message->tag);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->tag,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'tag'\n");
      return false;
    }
  }

  // Field name: params
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, PathIntegralParams
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->params))
    {
      return false;
    }
  }

  // Field name: stats
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, LapStats
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->stats))
    {
      return false;
    }
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__PathIntegralStats(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PathIntegralStats__ros_msg_type * ros_message = static_cast<const _PathIntegralStats__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name header

  current_alignment += get_serialized_size_std_msgs__msg__Header(
    &(ros_message->header), current_alignment);
  // field.name tag
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->tag.size + 1);
  // field.name params

  current_alignment += get_serialized_size_eufs_msgs__msg__PathIntegralParams(
    &(ros_message->params), current_alignment);
  // field.name stats

  current_alignment += get_serialized_size_eufs_msgs__msg__LapStats(
    &(ros_message->stats), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _PathIntegralStats__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__PathIntegralStats(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__PathIntegralStats(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: header
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_std_msgs__msg__Header(
        full_bounded, current_alignment);
    }
  }
  // member: tag
  {
    size_t array_size = 1;

    full_bounded = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: params
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_eufs_msgs__msg__PathIntegralParams(
        full_bounded, current_alignment);
    }
  }
  // member: stats
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_eufs_msgs__msg__LapStats(
        full_bounded, current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static size_t _PathIntegralStats__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__PathIntegralStats(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_PathIntegralStats = {
  "eufs_msgs::msg",
  "PathIntegralStats",
  _PathIntegralStats__cdr_serialize,
  _PathIntegralStats__cdr_deserialize,
  _PathIntegralStats__get_serialized_size,
  _PathIntegralStats__max_serialized_size
};

static rosidl_message_type_support_t _PathIntegralStats__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PathIntegralStats,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, PathIntegralStats)() {
  return &_PathIntegralStats__type_support;
}

#if defined(__cplusplus)
}
#endif
