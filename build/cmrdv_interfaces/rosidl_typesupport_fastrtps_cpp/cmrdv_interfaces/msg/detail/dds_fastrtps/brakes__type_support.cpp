// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from cmrdv_interfaces:msg/Brakes.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/brakes__rosidl_typesupport_fastrtps_cpp.hpp"
#include "cmrdv_interfaces/msg/detail/brakes__struct.hpp"

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
namespace builtin_interfaces
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const builtin_interfaces::msg::Time &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  builtin_interfaces::msg::Time &);
size_t get_serialized_size(
  const builtin_interfaces::msg::Time &,
  size_t current_alignment);
size_t
max_serialized_size_Time(
  bool & full_bounded,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace builtin_interfaces


namespace cmrdv_interfaces
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
cdr_serialize(
  const cmrdv_interfaces::msg::Brakes & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: braking
  cdr << (ros_message.braking ? true : false);
  // Member: last_fired
  builtin_interfaces::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.last_fired,
    cdr);
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  cmrdv_interfaces::msg::Brakes & ros_message)
{
  // Member: braking
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.braking = tmp ? true : false;
  }

  // Member: last_fired
  builtin_interfaces::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.last_fired);

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
get_serialized_size(
  const cmrdv_interfaces::msg::Brakes & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: braking
  {
    size_t item_size = sizeof(ros_message.braking);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: last_fired

  current_alignment +=
    builtin_interfaces::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.last_fired, current_alignment);

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
max_serialized_size_Brakes(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: braking
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: last_fired
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        builtin_interfaces::msg::typesupport_fastrtps_cpp::max_serialized_size_Time(
        full_bounded, current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static bool _Brakes__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const cmrdv_interfaces::msg::Brakes *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _Brakes__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<cmrdv_interfaces::msg::Brakes *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _Brakes__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const cmrdv_interfaces::msg::Brakes *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _Brakes__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_Brakes(full_bounded, 0);
}

static message_type_support_callbacks_t _Brakes__callbacks = {
  "cmrdv_interfaces::msg",
  "Brakes",
  _Brakes__cdr_serialize,
  _Brakes__cdr_deserialize,
  _Brakes__get_serialized_size,
  _Brakes__max_serialized_size
};

static rosidl_message_type_support_t _Brakes__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_Brakes__callbacks,
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
get_message_type_support_handle<cmrdv_interfaces::msg::Brakes>()
{
  return &cmrdv_interfaces::msg::typesupport_fastrtps_cpp::_Brakes__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, cmrdv_interfaces, msg, Brakes)() {
  return &cmrdv_interfaces::msg::typesupport_fastrtps_cpp::_Brakes__handle;
}

#ifdef __cplusplus
}
#endif
