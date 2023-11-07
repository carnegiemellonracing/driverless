// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from cmrdv_interfaces:msg/ControlAction.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/control_action__rosidl_typesupport_fastrtps_cpp.hpp"
#include "cmrdv_interfaces/msg/detail/control_action__struct.hpp"

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

namespace cmrdv_interfaces
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
cdr_serialize(
  const cmrdv_interfaces::msg::ControlAction & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: wheel_speed
  cdr << ros_message.wheel_speed;
  // Member: swangle
  cdr << ros_message.swangle;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  cmrdv_interfaces::msg::ControlAction & ros_message)
{
  // Member: wheel_speed
  cdr >> ros_message.wheel_speed;

  // Member: swangle
  cdr >> ros_message.swangle;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
get_serialized_size(
  const cmrdv_interfaces::msg::ControlAction & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: wheel_speed
  {
    size_t item_size = sizeof(ros_message.wheel_speed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: swangle
  {
    size_t item_size = sizeof(ros_message.swangle);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
max_serialized_size_ControlAction(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: wheel_speed
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: swangle
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static bool _ControlAction__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const cmrdv_interfaces::msg::ControlAction *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _ControlAction__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<cmrdv_interfaces::msg::ControlAction *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _ControlAction__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const cmrdv_interfaces::msg::ControlAction *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _ControlAction__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_ControlAction(full_bounded, 0);
}

static message_type_support_callbacks_t _ControlAction__callbacks = {
  "cmrdv_interfaces::msg",
  "ControlAction",
  _ControlAction__cdr_serialize,
  _ControlAction__cdr_deserialize,
  _ControlAction__get_serialized_size,
  _ControlAction__max_serialized_size
};

static rosidl_message_type_support_t _ControlAction__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_ControlAction__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace cmrdv_interfaces

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
get_message_type_support_handle<cmrdv_interfaces::msg::ControlAction>()
{
  return &cmrdv_interfaces::msg::typesupport_fastrtps_cpp::_ControlAction__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, cmrdv_interfaces, msg, ControlAction)() {
  return &cmrdv_interfaces::msg::typesupport_fastrtps_cpp::_ControlAction__handle;
}

#ifdef __cplusplus
}
#endif
