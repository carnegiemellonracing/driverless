// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/CanState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/can_state__rosidl_typesupport_fastrtps_cpp.hpp"
#include "eufs_msgs/msg/detail/can_state__struct.hpp"

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
  const eufs_msgs::msg::CanState & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: as_state
  cdr << ros_message.as_state;
  // Member: ami_state
  cdr << ros_message.ami_state;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  eufs_msgs::msg::CanState & ros_message)
{
  // Member: as_state
  cdr >> ros_message.as_state;

  // Member: ami_state
  cdr >> ros_message.ami_state;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
get_serialized_size(
  const eufs_msgs::msg::CanState & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: as_state
  {
    size_t item_size = sizeof(ros_message.as_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ami_state
  {
    size_t item_size = sizeof(ros_message.ami_state);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
max_serialized_size_CanState(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: as_state
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  // Member: ami_state
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint16_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint16_t));
  }

  return current_alignment - initial_alignment;
}

static bool _CanState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::CanState *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _CanState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<eufs_msgs::msg::CanState *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _CanState__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::CanState *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _CanState__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_CanState(full_bounded, 0);
}

static message_type_support_callbacks_t _CanState__callbacks = {
  "eufs_msgs::msg",
  "CanState",
  _CanState__cdr_serialize,
  _CanState__cdr_deserialize,
  _CanState__get_serialized_size,
  _CanState__max_serialized_size
};

static rosidl_message_type_support_t _CanState__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_CanState__callbacks,
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
get_message_type_support_handle<eufs_msgs::msg::CanState>()
{
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_CanState__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, eufs_msgs, msg, CanState)() {
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_CanState__handle;
}

#ifdef __cplusplus
}
#endif
