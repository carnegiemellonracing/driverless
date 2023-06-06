// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/ChassisState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/chassis_state__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/chassis_state__struct.h"
#include "eufs_msgs/msg/detail/chassis_state__functions.h"
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

#include "rosidl_runtime_c/string.h"  // front_brake_commander, steering_commander, throttle_commander
#include "rosidl_runtime_c/string_functions.h"  // front_brake_commander, steering_commander, throttle_commander
#include "std_msgs/msg/detail/header__functions.h"  // header

// forward declare type support functions
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


using _ChassisState__ros_msg_type = eufs_msgs__msg__ChassisState;

static bool _ChassisState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _ChassisState__ros_msg_type * ros_message = static_cast<const _ChassisState__ros_msg_type *>(untyped_ros_message);
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

  // Field name: throttle_relay_enabled
  {
    cdr << (ros_message->throttle_relay_enabled ? true : false);
  }

  // Field name: autonomous_enabled
  {
    cdr << (ros_message->autonomous_enabled ? true : false);
  }

  // Field name: runstop_motion_enabled
  {
    cdr << (ros_message->runstop_motion_enabled ? true : false);
  }

  // Field name: steering_commander
  {
    const rosidl_runtime_c__String * str = &ros_message->steering_commander;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: steering
  {
    cdr << ros_message->steering;
  }

  // Field name: throttle_commander
  {
    const rosidl_runtime_c__String * str = &ros_message->throttle_commander;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: throttle
  {
    cdr << ros_message->throttle;
  }

  // Field name: front_brake_commander
  {
    const rosidl_runtime_c__String * str = &ros_message->front_brake_commander;
    if (str->capacity == 0 || str->capacity <= str->size) {
      fprintf(stderr, "string capacity not greater than size\n");
      return false;
    }
    if (str->data[str->size] != '\0') {
      fprintf(stderr, "string not null-terminated\n");
      return false;
    }
    cdr << str->data;
  }

  // Field name: front_brake
  {
    cdr << ros_message->front_brake;
  }

  return true;
}

static bool _ChassisState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _ChassisState__ros_msg_type * ros_message = static_cast<_ChassisState__ros_msg_type *>(untyped_ros_message);
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

  // Field name: throttle_relay_enabled
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->throttle_relay_enabled = tmp ? true : false;
  }

  // Field name: autonomous_enabled
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->autonomous_enabled = tmp ? true : false;
  }

  // Field name: runstop_motion_enabled
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->runstop_motion_enabled = tmp ? true : false;
  }

  // Field name: steering_commander
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->steering_commander.data) {
      rosidl_runtime_c__String__init(&ros_message->steering_commander);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->steering_commander,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'steering_commander'\n");
      return false;
    }
  }

  // Field name: steering
  {
    cdr >> ros_message->steering;
  }

  // Field name: throttle_commander
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->throttle_commander.data) {
      rosidl_runtime_c__String__init(&ros_message->throttle_commander);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->throttle_commander,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'throttle_commander'\n");
      return false;
    }
  }

  // Field name: throttle
  {
    cdr >> ros_message->throttle;
  }

  // Field name: front_brake_commander
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->front_brake_commander.data) {
      rosidl_runtime_c__String__init(&ros_message->front_brake_commander);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->front_brake_commander,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'front_brake_commander'\n");
      return false;
    }
  }

  // Field name: front_brake
  {
    cdr >> ros_message->front_brake;
  }

  return true;
}  // NOLINT(readability/fn_size)

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__ChassisState(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _ChassisState__ros_msg_type * ros_message = static_cast<const _ChassisState__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name header

  current_alignment += get_serialized_size_std_msgs__msg__Header(
    &(ros_message->header), current_alignment);
  // field.name throttle_relay_enabled
  {
    size_t item_size = sizeof(ros_message->throttle_relay_enabled);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name autonomous_enabled
  {
    size_t item_size = sizeof(ros_message->autonomous_enabled);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name runstop_motion_enabled
  {
    size_t item_size = sizeof(ros_message->runstop_motion_enabled);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name steering_commander
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->steering_commander.size + 1);
  // field.name steering
  {
    size_t item_size = sizeof(ros_message->steering);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name throttle_commander
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->throttle_commander.size + 1);
  // field.name throttle
  {
    size_t item_size = sizeof(ros_message->throttle);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name front_brake_commander
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->front_brake_commander.size + 1);
  // field.name front_brake
  {
    size_t item_size = sizeof(ros_message->front_brake);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _ChassisState__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__ChassisState(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__ChassisState(
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
  // member: throttle_relay_enabled
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: autonomous_enabled
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: runstop_motion_enabled
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: steering_commander
  {
    size_t array_size = 1;

    full_bounded = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: steering
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: throttle_commander
  {
    size_t array_size = 1;

    full_bounded = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: throttle
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: front_brake_commander
  {
    size_t array_size = 1;

    full_bounded = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: front_brake
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static size_t _ChassisState__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__ChassisState(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_ChassisState = {
  "eufs_msgs::msg",
  "ChassisState",
  _ChassisState__cdr_serialize,
  _ChassisState__cdr_deserialize,
  _ChassisState__get_serialized_size,
  _ChassisState__max_serialized_size
};

static rosidl_message_type_support_t _ChassisState__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_ChassisState,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, ChassisState)() {
  return &_ChassisState__type_support;
}

#if defined(__cplusplus)
}
#endif
