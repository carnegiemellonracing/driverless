// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/ConeArrayWithCovariance.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/cone_array_with_covariance__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/cone_array_with_covariance__struct.h"
#include "eufs_msgs/msg/detail/cone_array_with_covariance__functions.h"
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

#include "eufs_msgs/msg/detail/cone_with_covariance__functions.h"  // big_orange_cones, blue_cones, orange_cones, unknown_color_cones, yellow_cones
#include "std_msgs/msg/detail/header__functions.h"  // header

// forward declare type support functions
size_t get_serialized_size_eufs_msgs__msg__ConeWithCovariance(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_eufs_msgs__msg__ConeWithCovariance(
  bool & full_bounded,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeWithCovariance)();
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


using _ConeArrayWithCovariance__ros_msg_type = eufs_msgs__msg__ConeArrayWithCovariance;

static bool _ConeArrayWithCovariance__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _ConeArrayWithCovariance__ros_msg_type * ros_message = static_cast<const _ConeArrayWithCovariance__ros_msg_type *>(untyped_ros_message);
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

  // Field name: blue_cones
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeWithCovariance
      )()->data);
    size_t size = ros_message->blue_cones.size;
    auto array_ptr = ros_message->blue_cones.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: yellow_cones
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeWithCovariance
      )()->data);
    size_t size = ros_message->yellow_cones.size;
    auto array_ptr = ros_message->yellow_cones.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: orange_cones
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeWithCovariance
      )()->data);
    size_t size = ros_message->orange_cones.size;
    auto array_ptr = ros_message->orange_cones.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: big_orange_cones
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeWithCovariance
      )()->data);
    size_t size = ros_message->big_orange_cones.size;
    auto array_ptr = ros_message->big_orange_cones.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  // Field name: unknown_color_cones
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeWithCovariance
      )()->data);
    size_t size = ros_message->unknown_color_cones.size;
    auto array_ptr = ros_message->unknown_color_cones.data;
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_serialize(
          &array_ptr[i], cdr))
      {
        return false;
      }
    }
  }

  return true;
}

static bool _ConeArrayWithCovariance__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _ConeArrayWithCovariance__ros_msg_type * ros_message = static_cast<_ConeArrayWithCovariance__ros_msg_type *>(untyped_ros_message);
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

  // Field name: blue_cones
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeWithCovariance
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->blue_cones.data) {
      eufs_msgs__msg__ConeWithCovariance__Sequence__fini(&ros_message->blue_cones);
    }
    if (!eufs_msgs__msg__ConeWithCovariance__Sequence__init(&ros_message->blue_cones, size)) {
      return "failed to create array for field 'blue_cones'";
    }
    auto array_ptr = ros_message->blue_cones.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: yellow_cones
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeWithCovariance
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->yellow_cones.data) {
      eufs_msgs__msg__ConeWithCovariance__Sequence__fini(&ros_message->yellow_cones);
    }
    if (!eufs_msgs__msg__ConeWithCovariance__Sequence__init(&ros_message->yellow_cones, size)) {
      return "failed to create array for field 'yellow_cones'";
    }
    auto array_ptr = ros_message->yellow_cones.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: orange_cones
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeWithCovariance
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->orange_cones.data) {
      eufs_msgs__msg__ConeWithCovariance__Sequence__fini(&ros_message->orange_cones);
    }
    if (!eufs_msgs__msg__ConeWithCovariance__Sequence__init(&ros_message->orange_cones, size)) {
      return "failed to create array for field 'orange_cones'";
    }
    auto array_ptr = ros_message->orange_cones.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: big_orange_cones
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeWithCovariance
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->big_orange_cones.data) {
      eufs_msgs__msg__ConeWithCovariance__Sequence__fini(&ros_message->big_orange_cones);
    }
    if (!eufs_msgs__msg__ConeWithCovariance__Sequence__init(&ros_message->big_orange_cones, size)) {
      return "failed to create array for field 'big_orange_cones'";
    }
    auto array_ptr = ros_message->big_orange_cones.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  // Field name: unknown_color_cones
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeWithCovariance
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->unknown_color_cones.data) {
      eufs_msgs__msg__ConeWithCovariance__Sequence__fini(&ros_message->unknown_color_cones);
    }
    if (!eufs_msgs__msg__ConeWithCovariance__Sequence__init(&ros_message->unknown_color_cones, size)) {
      return "failed to create array for field 'unknown_color_cones'";
    }
    auto array_ptr = ros_message->unknown_color_cones.data;
    for (size_t i = 0; i < size; ++i) {
      if (!callbacks->cdr_deserialize(
          cdr, &array_ptr[i]))
      {
        return false;
      }
    }
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__ConeArrayWithCovariance(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _ConeArrayWithCovariance__ros_msg_type * ros_message = static_cast<const _ConeArrayWithCovariance__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name header

  current_alignment += get_serialized_size_std_msgs__msg__Header(
    &(ros_message->header), current_alignment);
  // field.name blue_cones
  {
    size_t array_size = ros_message->blue_cones.size;
    auto array_ptr = ros_message->blue_cones.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_eufs_msgs__msg__ConeWithCovariance(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name yellow_cones
  {
    size_t array_size = ros_message->yellow_cones.size;
    auto array_ptr = ros_message->yellow_cones.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_eufs_msgs__msg__ConeWithCovariance(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name orange_cones
  {
    size_t array_size = ros_message->orange_cones.size;
    auto array_ptr = ros_message->orange_cones.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_eufs_msgs__msg__ConeWithCovariance(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name big_orange_cones
  {
    size_t array_size = ros_message->big_orange_cones.size;
    auto array_ptr = ros_message->big_orange_cones.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_eufs_msgs__msg__ConeWithCovariance(
        &array_ptr[index], current_alignment);
    }
  }
  // field.name unknown_color_cones
  {
    size_t array_size = ros_message->unknown_color_cones.size;
    auto array_ptr = ros_message->unknown_color_cones.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_eufs_msgs__msg__ConeWithCovariance(
        &array_ptr[index], current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static uint32_t _ConeArrayWithCovariance__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__ConeArrayWithCovariance(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__ConeArrayWithCovariance(
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
  // member: blue_cones
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_eufs_msgs__msg__ConeWithCovariance(
        full_bounded, current_alignment);
    }
  }
  // member: yellow_cones
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_eufs_msgs__msg__ConeWithCovariance(
        full_bounded, current_alignment);
    }
  }
  // member: orange_cones
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_eufs_msgs__msg__ConeWithCovariance(
        full_bounded, current_alignment);
    }
  }
  // member: big_orange_cones
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_eufs_msgs__msg__ConeWithCovariance(
        full_bounded, current_alignment);
    }
  }
  // member: unknown_color_cones
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_eufs_msgs__msg__ConeWithCovariance(
        full_bounded, current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static size_t _ConeArrayWithCovariance__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__ConeArrayWithCovariance(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_ConeArrayWithCovariance = {
  "eufs_msgs::msg",
  "ConeArrayWithCovariance",
  _ConeArrayWithCovariance__cdr_serialize,
  _ConeArrayWithCovariance__cdr_deserialize,
  _ConeArrayWithCovariance__get_serialized_size,
  _ConeArrayWithCovariance__max_serialized_size
};

static rosidl_message_type_support_t _ConeArrayWithCovariance__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_ConeArrayWithCovariance,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ConeArrayWithCovariance)() {
  return &_ConeArrayWithCovariance__type_support;
}

#if defined(__cplusplus)
}
#endif
