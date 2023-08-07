// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/Costmap.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/costmap__rosidl_typesupport_fastrtps_cpp.hpp"
#include "eufs_msgs/msg/detail/costmap__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions

namespace eufs_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
cdr_serialize(
  const eufs_msgs::msg::Costmap & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: x_bounds_min
  cdr << ros_message.x_bounds_min;
  // Member: x_bounds_max
  cdr << ros_message.x_bounds_max;
  // Member: y_bounds_min
  cdr << ros_message.y_bounds_min;
  // Member: y_bounds_max
  cdr << ros_message.y_bounds_max;
  // Member: pixels_per_meter
  cdr << ros_message.pixels_per_meter;
  // Member: channel0
  {
    cdr << ros_message.channel0;
  }
  // Member: channel1
  {
    cdr << ros_message.channel1;
  }
  // Member: channel2
  {
    cdr << ros_message.channel2;
  }
  // Member: channel3
  {
    cdr << ros_message.channel3;
  }
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  eufs_msgs::msg::Costmap & ros_message)
{
  // Member: x_bounds_min
  cdr >> ros_message.x_bounds_min;

  // Member: x_bounds_max
  cdr >> ros_message.x_bounds_max;

  // Member: y_bounds_min
  cdr >> ros_message.y_bounds_min;

  // Member: y_bounds_max
  cdr >> ros_message.y_bounds_max;

  // Member: pixels_per_meter
  cdr >> ros_message.pixels_per_meter;

  // Member: channel0
  {
    cdr >> ros_message.channel0;
  }

  // Member: channel1
  {
    cdr >> ros_message.channel1;
  }

  // Member: channel2
  {
    cdr >> ros_message.channel2;
  }

  // Member: channel3
  {
    cdr >> ros_message.channel3;
  }

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
get_serialized_size(
  const eufs_msgs::msg::Costmap & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: x_bounds_min
  {
    size_t item_size = sizeof(ros_message.x_bounds_min);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: x_bounds_max
  {
    size_t item_size = sizeof(ros_message.x_bounds_max);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: y_bounds_min
  {
    size_t item_size = sizeof(ros_message.y_bounds_min);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: y_bounds_max
  {
    size_t item_size = sizeof(ros_message.y_bounds_max);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: pixels_per_meter
  {
    size_t item_size = sizeof(ros_message.pixels_per_meter);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: channel0
  {
    size_t array_size = ros_message.channel0.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    size_t item_size = sizeof(ros_message.channel0[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: channel1
  {
    size_t array_size = ros_message.channel1.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    size_t item_size = sizeof(ros_message.channel1[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: channel2
  {
    size_t array_size = ros_message.channel2.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    size_t item_size = sizeof(ros_message.channel2[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: channel3
  {
    size_t array_size = ros_message.channel3.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);
    size_t item_size = sizeof(ros_message.channel3[0]);
    current_alignment += array_size * item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
max_serialized_size_Costmap(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: x_bounds_min
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: x_bounds_max
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: y_bounds_min
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: y_bounds_max
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: pixels_per_meter
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: channel0
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: channel1
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: channel2
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: channel3
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

static bool _Costmap__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::Costmap *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _Costmap__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<eufs_msgs::msg::Costmap *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _Costmap__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::Costmap *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _Costmap__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_Costmap(full_bounded, 0);
}

static message_type_support_callbacks_t _Costmap__callbacks = {
  "eufs_msgs::msg",
  "Costmap",
  _Costmap__cdr_serialize,
  _Costmap__cdr_deserialize,
  _Costmap__get_serialized_size,
  _Costmap__max_serialized_size
};

static rosidl_message_type_support_t _Costmap__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_Costmap__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace eufs_msgs

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
get_message_type_support_handle<eufs_msgs::msg::Costmap>()
{
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_Costmap__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, eufs_msgs, msg, Costmap)() {
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_Costmap__handle;
}

#ifdef __cplusplus
}
#endif
