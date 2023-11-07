// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from cmrdv_interfaces:msg/PairROT.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/pair_rot__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "cmrdv_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "cmrdv_interfaces/msg/detail/pair_rot__struct.h"
#include "cmrdv_interfaces/msg/detail/pair_rot__functions.h"
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

#include "cmrdv_interfaces/msg/detail/car_rot__functions.h"  // far, near
#include "std_msgs/msg/detail/header__functions.h"  // header

// forward declare type support functions
size_t get_serialized_size_cmrdv_interfaces__msg__CarROT(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_cmrdv_interfaces__msg__CarROT(
  bool & full_bounded,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, cmrdv_interfaces, msg, CarROT)();
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t get_serialized_size_std_msgs__msg__Header(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t max_serialized_size_std_msgs__msg__Header(
  bool & full_bounded,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, std_msgs, msg, Header)();


using _PairROT__ros_msg_type = cmrdv_interfaces__msg__PairROT;

static bool _PairROT__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _PairROT__ros_msg_type * ros_message = static_cast<const _PairROT__ros_msg_type *>(untyped_ros_message);
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

  // Field name: near
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, cmrdv_interfaces, msg, CarROT
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->near, cdr))
    {
      return false;
    }
  }

  // Field name: far
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, cmrdv_interfaces, msg, CarROT
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->far, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _PairROT__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _PairROT__ros_msg_type * ros_message = static_cast<_PairROT__ros_msg_type *>(untyped_ros_message);
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

  // Field name: near
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, cmrdv_interfaces, msg, CarROT
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->near))
    {
      return false;
    }
  }

  // Field name: far
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, cmrdv_interfaces, msg, CarROT
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->far))
    {
      return false;
    }
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_cmrdv_interfaces
size_t get_serialized_size_cmrdv_interfaces__msg__PairROT(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PairROT__ros_msg_type * ros_message = static_cast<const _PairROT__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name header

  current_alignment += get_serialized_size_std_msgs__msg__Header(
    &(ros_message->header), current_alignment);
  // field.name near

  current_alignment += get_serialized_size_cmrdv_interfaces__msg__CarROT(
    &(ros_message->near), current_alignment);
  // field.name far

  current_alignment += get_serialized_size_cmrdv_interfaces__msg__CarROT(
    &(ros_message->far), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _PairROT__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_cmrdv_interfaces__msg__PairROT(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_cmrdv_interfaces
size_t max_serialized_size_cmrdv_interfaces__msg__PairROT(
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
  // member: near
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_cmrdv_interfaces__msg__CarROT(
        full_bounded, current_alignment);
    }
  }
  // member: far
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_cmrdv_interfaces__msg__CarROT(
        full_bounded, current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static size_t _PairROT__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_cmrdv_interfaces__msg__PairROT(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_PairROT = {
  "cmrdv_interfaces::msg",
  "PairROT",
  _PairROT__cdr_serialize,
  _PairROT__cdr_deserialize,
  _PairROT__get_serialized_size,
  _PairROT__max_serialized_size
};

static rosidl_message_type_support_t _PairROT__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PairROT,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, cmrdv_interfaces, msg, PairROT)() {
  return &_PairROT__type_support;
}

#if defined(__cplusplus)
}
#endif
