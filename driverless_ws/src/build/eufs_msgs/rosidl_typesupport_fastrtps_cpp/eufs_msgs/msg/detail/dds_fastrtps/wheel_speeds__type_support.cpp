// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/WheelSpeeds.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/wheel_speeds__rosidl_typesupport_fastrtps_cpp.hpp"
#include "eufs_msgs/msg/detail/wheel_speeds__struct.hpp"

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
  const eufs_msgs::msg::WheelSpeeds & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: steering
  cdr << ros_message.steering;
  // Member: lf_speed
  cdr << ros_message.lf_speed;
  // Member: rf_speed
  cdr << ros_message.rf_speed;
  // Member: lb_speed
  cdr << ros_message.lb_speed;
  // Member: rb_speed
  cdr << ros_message.rb_speed;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  eufs_msgs::msg::WheelSpeeds & ros_message)
{
  // Member: steering
  cdr >> ros_message.steering;

  // Member: lf_speed
  cdr >> ros_message.lf_speed;

  // Member: rf_speed
  cdr >> ros_message.rf_speed;

  // Member: lb_speed
  cdr >> ros_message.lb_speed;

  // Member: rb_speed
  cdr >> ros_message.rb_speed;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
get_serialized_size(
  const eufs_msgs::msg::WheelSpeeds & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: steering
  {
    size_t item_size = sizeof(ros_message.steering);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: lf_speed
  {
    size_t item_size = sizeof(ros_message.lf_speed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: rf_speed
  {
    size_t item_size = sizeof(ros_message.rf_speed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: lb_speed
  {
    size_t item_size = sizeof(ros_message.lb_speed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: rb_speed
  {
    size_t item_size = sizeof(ros_message.rb_speed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
max_serialized_size_WheelSpeeds(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: steering
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: lf_speed
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: rf_speed
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: lb_speed
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Member: rb_speed
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  return current_alignment - initial_alignment;
}

static bool _WheelSpeeds__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::WheelSpeeds *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _WheelSpeeds__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<eufs_msgs::msg::WheelSpeeds *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _WheelSpeeds__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::WheelSpeeds *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _WheelSpeeds__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_WheelSpeeds(full_bounded, 0);
}

static message_type_support_callbacks_t _WheelSpeeds__callbacks = {
  "eufs_msgs::msg",
  "WheelSpeeds",
  _WheelSpeeds__cdr_serialize,
  _WheelSpeeds__cdr_deserialize,
  _WheelSpeeds__get_serialized_size,
  _WheelSpeeds__max_serialized_size
};

static rosidl_message_type_support_t _WheelSpeeds__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_WheelSpeeds__callbacks,
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
get_message_type_support_handle<eufs_msgs::msg::WheelSpeeds>()
{
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_WheelSpeeds__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, eufs_msgs, msg, WheelSpeeds)() {
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_WheelSpeeds__handle;
}

#ifdef __cplusplus
}
#endif
