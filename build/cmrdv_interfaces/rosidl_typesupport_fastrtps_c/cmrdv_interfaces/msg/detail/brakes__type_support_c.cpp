// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from cmrdv_interfaces:msg/Brakes.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/brakes__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "cmrdv_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "cmrdv_interfaces/msg/detail/brakes__struct.h"
#include "cmrdv_interfaces/msg/detail/brakes__functions.h"
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

#include "builtin_interfaces/msg/detail/time__functions.h"  // last_fired

// forward declare type support functions
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t get_serialized_size_builtin_interfaces__msg__Time(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t max_serialized_size_builtin_interfaces__msg__Time(
  bool & full_bounded,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time)();


using _Brakes__ros_msg_type = cmrdv_interfaces__msg__Brakes;

static bool _Brakes__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _Brakes__ros_msg_type * ros_message = static_cast<const _Brakes__ros_msg_type *>(untyped_ros_message);
  // Field name: braking
  {
    cdr << (ros_message->braking ? true : false);
  }

  // Field name: last_fired
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->last_fired, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _Brakes__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _Brakes__ros_msg_type * ros_message = static_cast<_Brakes__ros_msg_type *>(untyped_ros_message);
  // Field name: braking
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->braking = tmp ? true : false;
  }

  // Field name: last_fired
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, builtin_interfaces, msg, Time
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->last_fired))
    {
      return false;
    }
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_cmrdv_interfaces
size_t get_serialized_size_cmrdv_interfaces__msg__Brakes(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _Brakes__ros_msg_type * ros_message = static_cast<const _Brakes__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name braking
  {
    size_t item_size = sizeof(ros_message->braking);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name last_fired

  current_alignment += get_serialized_size_builtin_interfaces__msg__Time(
    &(ros_message->last_fired), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _Brakes__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_cmrdv_interfaces__msg__Brakes(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_cmrdv_interfaces
size_t max_serialized_size_cmrdv_interfaces__msg__Brakes(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: braking
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: last_fired
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_builtin_interfaces__msg__Time(
        full_bounded, current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static size_t _Brakes__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_cmrdv_interfaces__msg__Brakes(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_Brakes = {
  "cmrdv_interfaces::msg",
  "Brakes",
  _Brakes__cdr_serialize,
  _Brakes__cdr_deserialize,
  _Brakes__get_serialized_size,
  _Brakes__max_serialized_size
};

static rosidl_message_type_support_t _Brakes__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_Brakes,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, cmrdv_interfaces, msg, Brakes)() {
  return &_Brakes__type_support;
}

#if defined(__cplusplus)
}
#endif
