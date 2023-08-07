// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/WaypointArrayStamped.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/waypoint_array_stamped__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/waypoint_array_stamped__struct.h"
#include "eufs_msgs/msg/detail/waypoint_array_stamped__functions.h"
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

#include "eufs_msgs/msg/detail/waypoint__functions.h"  // waypoints
#include "std_msgs/msg/detail/header__functions.h"  // header

// forward declare type support functions
size_t get_serialized_size_eufs_msgs__msg__Waypoint(
  const void * untyped_ros_message,
  size_t current_alignment);

size_t max_serialized_size_eufs_msgs__msg__Waypoint(
  bool & full_bounded,
  size_t current_alignment);

const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, Waypoint)();
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


using _WaypointArrayStamped__ros_msg_type = eufs_msgs__msg__WaypointArrayStamped;

static bool _WaypointArrayStamped__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _WaypointArrayStamped__ros_msg_type * ros_message = static_cast<const _WaypointArrayStamped__ros_msg_type *>(untyped_ros_message);
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

  // Field name: waypoints
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, Waypoint
      )()->data);
    size_t size = ros_message->waypoints.size;
    auto array_ptr = ros_message->waypoints.data;
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

static bool _WaypointArrayStamped__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _WaypointArrayStamped__ros_msg_type * ros_message = static_cast<_WaypointArrayStamped__ros_msg_type *>(untyped_ros_message);
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

  // Field name: waypoints
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, eufs_msgs, msg, Waypoint
      )()->data);
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    if (ros_message->waypoints.data) {
      eufs_msgs__msg__Waypoint__Sequence__fini(&ros_message->waypoints);
    }
    if (!eufs_msgs__msg__Waypoint__Sequence__init(&ros_message->waypoints, size)) {
      return "failed to create array for field 'waypoints'";
    }
    auto array_ptr = ros_message->waypoints.data;
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
size_t get_serialized_size_eufs_msgs__msg__WaypointArrayStamped(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _WaypointArrayStamped__ros_msg_type * ros_message = static_cast<const _WaypointArrayStamped__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name header

  current_alignment += get_serialized_size_std_msgs__msg__Header(
    &(ros_message->header), current_alignment);
  // field.name waypoints
  {
    size_t array_size = ros_message->waypoints.size;
    auto array_ptr = ros_message->waypoints.data;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += get_serialized_size_eufs_msgs__msg__Waypoint(
        &array_ptr[index], current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static uint32_t _WaypointArrayStamped__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__WaypointArrayStamped(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__WaypointArrayStamped(
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
  // member: waypoints
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_eufs_msgs__msg__Waypoint(
        full_bounded, current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static size_t _WaypointArrayStamped__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__WaypointArrayStamped(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_WaypointArrayStamped = {
  "eufs_msgs::msg",
  "WaypointArrayStamped",
  _WaypointArrayStamped__cdr_serialize,
  _WaypointArrayStamped__cdr_deserialize,
  _WaypointArrayStamped__get_serialized_size,
  _WaypointArrayStamped__max_serialized_size
};

static rosidl_message_type_support_t _WaypointArrayStamped__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_WaypointArrayStamped,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, WaypointArrayStamped)() {
  return &_WaypointArrayStamped__type_support;
}

#if defined(__cplusplus)
}
#endif
