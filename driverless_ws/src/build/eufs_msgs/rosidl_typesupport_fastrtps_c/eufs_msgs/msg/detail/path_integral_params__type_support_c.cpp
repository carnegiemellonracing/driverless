// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/PathIntegralParams.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/path_integral_params__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/path_integral_params__struct.h"
#include "eufs_msgs/msg/detail/path_integral_params__functions.h"
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

#include "rosidl_runtime_c/string.h"  // map_path
#include "rosidl_runtime_c/string_functions.h"  // map_path

// forward declare type support functions


using _PathIntegralParams__ros_msg_type = eufs_msgs__msg__PathIntegralParams;

static bool _PathIntegralParams__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _PathIntegralParams__ros_msg_type * ros_message = static_cast<const _PathIntegralParams__ros_msg_type *>(untyped_ros_message);
  // Field name: hz
  {
    cdr << ros_message->hz;
  }

  // Field name: num_timesteps
  {
    cdr << ros_message->num_timesteps;
  }

  // Field name: num_iters
  {
    cdr << ros_message->num_iters;
  }

  // Field name: gamma
  {
    cdr << ros_message->gamma;
  }

  // Field name: init_steering
  {
    cdr << ros_message->init_steering;
  }

  // Field name: init_throttle
  {
    cdr << ros_message->init_throttle;
  }

  // Field name: steering_var
  {
    cdr << ros_message->steering_var;
  }

  // Field name: throttle_var
  {
    cdr << ros_message->throttle_var;
  }

  // Field name: max_throttle
  {
    cdr << ros_message->max_throttle;
  }

  // Field name: speed_coefficient
  {
    cdr << ros_message->speed_coefficient;
  }

  // Field name: track_coefficient
  {
    cdr << ros_message->track_coefficient;
  }

  // Field name: max_slip_angle
  {
    cdr << ros_message->max_slip_angle;
  }

  // Field name: track_slop
  {
    cdr << ros_message->track_slop;
  }

  // Field name: crash_coeff
  {
    cdr << ros_message->crash_coeff;
  }

  // Field name: map_path
  {
    const rosidl_runtime_c__String * str = &ros_message->map_path;
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

  // Field name: desired_speed
  {
    cdr << ros_message->desired_speed;
  }

  return true;
}

static bool _PathIntegralParams__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _PathIntegralParams__ros_msg_type * ros_message = static_cast<_PathIntegralParams__ros_msg_type *>(untyped_ros_message);
  // Field name: hz
  {
    cdr >> ros_message->hz;
  }

  // Field name: num_timesteps
  {
    cdr >> ros_message->num_timesteps;
  }

  // Field name: num_iters
  {
    cdr >> ros_message->num_iters;
  }

  // Field name: gamma
  {
    cdr >> ros_message->gamma;
  }

  // Field name: init_steering
  {
    cdr >> ros_message->init_steering;
  }

  // Field name: init_throttle
  {
    cdr >> ros_message->init_throttle;
  }

  // Field name: steering_var
  {
    cdr >> ros_message->steering_var;
  }

  // Field name: throttle_var
  {
    cdr >> ros_message->throttle_var;
  }

  // Field name: max_throttle
  {
    cdr >> ros_message->max_throttle;
  }

  // Field name: speed_coefficient
  {
    cdr >> ros_message->speed_coefficient;
  }

  // Field name: track_coefficient
  {
    cdr >> ros_message->track_coefficient;
  }

  // Field name: max_slip_angle
  {
    cdr >> ros_message->max_slip_angle;
  }

  // Field name: track_slop
  {
    cdr >> ros_message->track_slop;
  }

  // Field name: crash_coeff
  {
    cdr >> ros_message->crash_coeff;
  }

  // Field name: map_path
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->map_path.data) {
      rosidl_runtime_c__String__init(&ros_message->map_path);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->map_path,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'map_path'\n");
      return false;
    }
  }

  // Field name: desired_speed
  {
    cdr >> ros_message->desired_speed;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__PathIntegralParams(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _PathIntegralParams__ros_msg_type * ros_message = static_cast<const _PathIntegralParams__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name hz
  {
    size_t item_size = sizeof(ros_message->hz);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name num_timesteps
  {
    size_t item_size = sizeof(ros_message->num_timesteps);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name num_iters
  {
    size_t item_size = sizeof(ros_message->num_iters);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name gamma
  {
    size_t item_size = sizeof(ros_message->gamma);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name init_steering
  {
    size_t item_size = sizeof(ros_message->init_steering);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name init_throttle
  {
    size_t item_size = sizeof(ros_message->init_throttle);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name steering_var
  {
    size_t item_size = sizeof(ros_message->steering_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name throttle_var
  {
    size_t item_size = sizeof(ros_message->throttle_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name max_throttle
  {
    size_t item_size = sizeof(ros_message->max_throttle);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name speed_coefficient
  {
    size_t item_size = sizeof(ros_message->speed_coefficient);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name track_coefficient
  {
    size_t item_size = sizeof(ros_message->track_coefficient);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name max_slip_angle
  {
    size_t item_size = sizeof(ros_message->max_slip_angle);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name track_slop
  {
    size_t item_size = sizeof(ros_message->track_slop);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name crash_coeff
  {
    size_t item_size = sizeof(ros_message->crash_coeff);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name map_path
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->map_path.size + 1);
  // field.name desired_speed
  {
    size_t item_size = sizeof(ros_message->desired_speed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _PathIntegralParams__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__PathIntegralParams(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__PathIntegralParams(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: hz
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: num_timesteps
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: num_iters
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: gamma
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: init_steering
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: init_throttle
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: steering_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: throttle_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: max_throttle
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: speed_coefficient
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: track_coefficient
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: max_slip_angle
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: track_slop
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: crash_coeff
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: map_path
  {
    size_t array_size = 1;

    full_bounded = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: desired_speed
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static size_t _PathIntegralParams__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__PathIntegralParams(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_PathIntegralParams = {
  "eufs_msgs::msg",
  "PathIntegralParams",
  _PathIntegralParams__cdr_serialize,
  _PathIntegralParams__cdr_deserialize,
  _PathIntegralParams__get_serialized_size,
  _PathIntegralParams__max_serialized_size
};

static rosidl_message_type_support_t _PathIntegralParams__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_PathIntegralParams,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, PathIntegralParams)() {
  return &_PathIntegralParams__type_support;
}

#if defined(__cplusplus)
}
#endif
