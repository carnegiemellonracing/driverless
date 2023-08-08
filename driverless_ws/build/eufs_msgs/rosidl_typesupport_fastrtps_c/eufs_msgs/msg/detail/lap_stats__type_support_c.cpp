// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/LapStats.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/lap_stats__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/lap_stats__struct.h"
#include "eufs_msgs/msg/detail/lap_stats__functions.h"
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

#include "std_msgs/msg/detail/header__functions.h"  // header

// forward declare type support functions
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


using _LapStats__ros_msg_type = eufs_msgs__msg__LapStats;

static bool _LapStats__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _LapStats__ros_msg_type * ros_message = static_cast<const _LapStats__ros_msg_type *>(untyped_ros_message);
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

  // Field name: lap_number
  {
    cdr << ros_message->lap_number;
  }

  // Field name: lap_time
  {
    cdr << ros_message->lap_time;
  }

  // Field name: avg_speed
  {
    cdr << ros_message->avg_speed;
  }

  // Field name: max_speed
  {
    cdr << ros_message->max_speed;
  }

  // Field name: speed_var
  {
    cdr << ros_message->speed_var;
  }

  // Field name: max_slip
  {
    cdr << ros_message->max_slip;
  }

  // Field name: max_lateral_accel
  {
    cdr << ros_message->max_lateral_accel;
  }

  // Field name: normalized_deviation_mse
  {
    cdr << ros_message->normalized_deviation_mse;
  }

  // Field name: deviation_var
  {
    cdr << ros_message->deviation_var;
  }

  // Field name: max_deviation
  {
    cdr << ros_message->max_deviation;
  }

  return true;
}

static bool _LapStats__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _LapStats__ros_msg_type * ros_message = static_cast<_LapStats__ros_msg_type *>(untyped_ros_message);
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

  // Field name: lap_number
  {
    cdr >> ros_message->lap_number;
  }

  // Field name: lap_time
  {
    cdr >> ros_message->lap_time;
  }

  // Field name: avg_speed
  {
    cdr >> ros_message->avg_speed;
  }

  // Field name: max_speed
  {
    cdr >> ros_message->max_speed;
  }

  // Field name: speed_var
  {
    cdr >> ros_message->speed_var;
  }

  // Field name: max_slip
  {
    cdr >> ros_message->max_slip;
  }

  // Field name: max_lateral_accel
  {
    cdr >> ros_message->max_lateral_accel;
  }

  // Field name: normalized_deviation_mse
  {
    cdr >> ros_message->normalized_deviation_mse;
  }

  // Field name: deviation_var
  {
    cdr >> ros_message->deviation_var;
  }

  // Field name: max_deviation
  {
    cdr >> ros_message->max_deviation;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__LapStats(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _LapStats__ros_msg_type * ros_message = static_cast<const _LapStats__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name header

  current_alignment += get_serialized_size_std_msgs__msg__Header(
    &(ros_message->header), current_alignment);
  // field.name lap_number
  {
    size_t item_size = sizeof(ros_message->lap_number);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name lap_time
  {
    size_t item_size = sizeof(ros_message->lap_time);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name avg_speed
  {
    size_t item_size = sizeof(ros_message->avg_speed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name max_speed
  {
    size_t item_size = sizeof(ros_message->max_speed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name speed_var
  {
    size_t item_size = sizeof(ros_message->speed_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name max_slip
  {
    size_t item_size = sizeof(ros_message->max_slip);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name max_lateral_accel
  {
    size_t item_size = sizeof(ros_message->max_lateral_accel);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name normalized_deviation_mse
  {
    size_t item_size = sizeof(ros_message->normalized_deviation_mse);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name deviation_var
  {
    size_t item_size = sizeof(ros_message->deviation_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name max_deviation
  {
    size_t item_size = sizeof(ros_message->max_deviation);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _LapStats__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__LapStats(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__LapStats(
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
  // member: lap_number
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: lap_time
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: avg_speed
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: max_speed
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: speed_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: max_slip
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: max_lateral_accel
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: normalized_deviation_mse
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: deviation_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: max_deviation
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static size_t _LapStats__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__LapStats(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_LapStats = {
  "eufs_msgs::msg",
  "LapStats",
  _LapStats__cdr_serialize,
  _LapStats__cdr_deserialize,
  _LapStats__get_serialized_size,
  _LapStats__max_serialized_size
};

static rosidl_message_type_support_t _LapStats__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_LapStats,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, LapStats)() {
  return &_LapStats__type_support;
}

#if defined(__cplusplus)
}
#endif
