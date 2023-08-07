// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/Costmap.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/costmap__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/costmap__struct.h"
#include "eufs_msgs/msg/detail/costmap__functions.h"
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

#include "rosidl_runtime_c/primitives_sequence.h"  // channel0, channel1, channel2, channel3
#include "rosidl_runtime_c/primitives_sequence_functions.h"  // channel0, channel1, channel2, channel3

// forward declare type support functions


using _Costmap__ros_msg_type = eufs_msgs__msg__Costmap;

static bool _Costmap__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _Costmap__ros_msg_type * ros_message = static_cast<const _Costmap__ros_msg_type *>(untyped_ros_message);
  // Field name: x_bounds_min
  {
    cdr << ros_message->x_bounds_min;
  }

  // Field name: x_bounds_max
  {
    cdr << ros_message->x_bounds_max;
  }

  // Field name: y_bounds_min
  {
    cdr << ros_message->y_bounds_min;
  }

  // Field name: y_bounds_max
  {
    cdr << ros_message->y_bounds_max;
  }

  // Field name: pixels_per_meter
  {
    cdr << ros_message->pixels_per_meter;
  }

  // Field name: channel0
  {
    size_t size = ros_message->channel0.size;
    auto array_ptr = ros_message->channel0.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: channel1
  {
    size_t size = ros_message->channel1.size;
    auto array_ptr = ros_message->channel1.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: channel2
  {
    size_t size = ros_message->channel2.size;
    auto array_ptr = ros_message->channel2.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  // Field name: channel3
  {
    size_t size = ros_message->channel3.size;
    auto array_ptr = ros_message->channel3.data;
    cdr << static_cast<uint32_t>(size);
    cdr.serializeArray(array_ptr, size);
  }

  return true;
}

static bool _Costmap__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _Costmap__ros_msg_type * ros_message = static_cast<_Costmap__ros_msg_type *>(untyped_ros_message);
  // Field name: x_bounds_min
  {
    cdr >> ros_message->x_bounds_min;
  }

  // Field name: x_bounds_max
  {
    cdr >> ros_message->x_bounds_max;
  }

  // Field name: y_bounds_min
  {
    cdr >> ros_message->y_bounds_min;
  }

  // Field name: y_bounds_max
  {
    cdr >> ros_message->y_bounds_max;
  }

  // Field name: pixels_per_meter
  {
    cdr >> ros_message->pixels_per_meter;
  }

  // Field name: channel0
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->channel0.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->channel0);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->channel0, size)) {
      return "failed to create array for field 'channel0'";
    }
    auto array_ptr = ros_message->channel0.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: channel1
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->channel1.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->channel1);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->channel1, size)) {
      return "failed to create array for field 'channel1'";
    }
    auto array_ptr = ros_message->channel1.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: channel2
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->channel2.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->channel2);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->channel2, size)) {
      return "failed to create array for field 'channel2'";
    }
    auto array_ptr = ros_message->channel2.data;
    cdr.deserializeArray(array_ptr, size);
  }

  // Field name: channel3
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->channel3.data) {
      rosidl_runtime_c__float__Sequence__fini(&ros_message->channel3);
    }
    if (!rosidl_runtime_c__float__Sequence__init(&ros_message->channel3, size)) {
      return "failed to create array for field 'channel3'";
    }
    auto array_ptr = ros_message->channel3.data;
    cdr.deserializeArray(array_ptr, size);
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__Costmap(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _Costmap__ros_msg_type * ros_message = static_cast<const _Costmap__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name x_bounds_min
  {
    size_t item_size = sizeof(ros_message->x_bounds_min);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name x_bounds_max
  {
    size_t item_size = sizeof(ros_message->x_bounds_max);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name y_bounds_min
  {
    size_t item_size = sizeof(ros_message->y_bounds_min);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name y_bounds_max
  {
    size_t item_size = sizeof(ros_message->y_bounds_max);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name pixels_per_meter
  {
    size_t item_size = sizeof(ros_message->pixels_per_meter);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name channel0
  {
    size_t array_size = ros_message->channel0.size;
    auto array_ptr = ros_message->channel0.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name channel1
  {
    size_t array_size = ros_message->channel1.size;
    auto array_ptr = ros_message->channel1.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name channel2
  {
    size_t array_size = ros_message->channel2.size;
    auto array_ptr = ros_message->channel2.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name channel3
  {
    size_t array_size = ros_message->channel3.size;
    auto array_ptr = ros_message->channel3.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    (void)array_ptr;
    size_t item_size = sizeof(array_ptr[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _Costmap__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__Costmap(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__Costmap(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: x_bounds_min
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: x_bounds_max
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: y_bounds_min
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: y_bounds_max
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: pixels_per_meter
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: channel0
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: channel1
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: channel2
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }
  // member: channel3
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  return current_alignment - initial_alignment;
}

static size_t _Costmap__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__Costmap(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_Costmap = {
  "eufs_msgs::msg",
  "Costmap",
  _Costmap__cdr_serialize,
  _Costmap__cdr_deserialize,
  _Costmap__get_serialized_size,
  _Costmap__max_serialized_size
};

static rosidl_message_type_support_t _Costmap__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_Costmap,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, Costmap)() {
  return &_Costmap__type_support;
}

#if defined(__cplusplus)
}
#endif
