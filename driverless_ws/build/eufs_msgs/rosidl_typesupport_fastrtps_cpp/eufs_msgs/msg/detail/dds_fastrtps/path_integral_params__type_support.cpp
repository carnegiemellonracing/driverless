// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/PathIntegralParams.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/path_integral_params__rosidl_typesupport_fastrtps_cpp.hpp"
#include "eufs_msgs/msg/detail/path_integral_params__struct.hpp"

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
  const eufs_msgs::msg::PathIntegralParams & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: hz
  cdr << ros_message.hz;
  // Member: num_timesteps
  cdr << ros_message.num_timesteps;
  // Member: num_iters
  cdr << ros_message.num_iters;
  // Member: gamma
  cdr << ros_message.gamma;
  // Member: init_steering
  cdr << ros_message.init_steering;
  // Member: init_throttle
  cdr << ros_message.init_throttle;
  // Member: steering_var
  cdr << ros_message.steering_var;
  // Member: throttle_var
  cdr << ros_message.throttle_var;
  // Member: max_throttle
  cdr << ros_message.max_throttle;
  // Member: speed_coefficient
  cdr << ros_message.speed_coefficient;
  // Member: track_coefficient
  cdr << ros_message.track_coefficient;
  // Member: max_slip_angle
  cdr << ros_message.max_slip_angle;
  // Member: track_slop
  cdr << ros_message.track_slop;
  // Member: crash_coeff
  cdr << ros_message.crash_coeff;
  // Member: map_path
  cdr << ros_message.map_path;
  // Member: desired_speed
  cdr << ros_message.desired_speed;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  eufs_msgs::msg::PathIntegralParams & ros_message)
{
  // Member: hz
  cdr >> ros_message.hz;

  // Member: num_timesteps
  cdr >> ros_message.num_timesteps;

  // Member: num_iters
  cdr >> ros_message.num_iters;

  // Member: gamma
  cdr >> ros_message.gamma;

  // Member: init_steering
  cdr >> ros_message.init_steering;

  // Member: init_throttle
  cdr >> ros_message.init_throttle;

  // Member: steering_var
  cdr >> ros_message.steering_var;

  // Member: throttle_var
  cdr >> ros_message.throttle_var;

  // Member: max_throttle
  cdr >> ros_message.max_throttle;

  // Member: speed_coefficient
  cdr >> ros_message.speed_coefficient;

  // Member: track_coefficient
  cdr >> ros_message.track_coefficient;

  // Member: max_slip_angle
  cdr >> ros_message.max_slip_angle;

  // Member: track_slop
  cdr >> ros_message.track_slop;

  // Member: crash_coeff
  cdr >> ros_message.crash_coeff;

  // Member: map_path
  cdr >> ros_message.map_path;

  // Member: desired_speed
  cdr >> ros_message.desired_speed;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
get_serialized_size(
  const eufs_msgs::msg::PathIntegralParams & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: hz
  {
    size_t item_size = sizeof(ros_message.hz);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: num_timesteps
  {
    size_t item_size = sizeof(ros_message.num_timesteps);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: num_iters
  {
    size_t item_size = sizeof(ros_message.num_iters);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: gamma
  {
    size_t item_size = sizeof(ros_message.gamma);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: init_steering
  {
    size_t item_size = sizeof(ros_message.init_steering);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: init_throttle
  {
    size_t item_size = sizeof(ros_message.init_throttle);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: steering_var
  {
    size_t item_size = sizeof(ros_message.steering_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: throttle_var
  {
    size_t item_size = sizeof(ros_message.throttle_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: max_throttle
  {
    size_t item_size = sizeof(ros_message.max_throttle);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: speed_coefficient
  {
    size_t item_size = sizeof(ros_message.speed_coefficient);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: track_coefficient
  {
    size_t item_size = sizeof(ros_message.track_coefficient);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: max_slip_angle
  {
    size_t item_size = sizeof(ros_message.max_slip_angle);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: track_slop
  {
    size_t item_size = sizeof(ros_message.track_slop);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: crash_coeff
  {
    size_t item_size = sizeof(ros_message.crash_coeff);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: map_path
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message.map_path.size() + 1);
  // Member: desired_speed
  {
    size_t item_size = sizeof(ros_message.desired_speed);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
max_serialized_size_PathIntegralParams(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: hz
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: num_timesteps
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: num_iters
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: gamma
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: init_steering
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: init_throttle
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: steering_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: throttle_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: max_throttle
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: speed_coefficient
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: track_coefficient
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: max_slip_angle
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: track_slop
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: crash_coeff
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: map_path
  {
    size_t array_size = 1;

    full_bounded = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }

  // Member: desired_speed
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static bool _PathIntegralParams__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::PathIntegralParams *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _PathIntegralParams__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<eufs_msgs::msg::PathIntegralParams *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _PathIntegralParams__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::PathIntegralParams *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _PathIntegralParams__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_PathIntegralParams(full_bounded, 0);
}

static message_type_support_callbacks_t _PathIntegralParams__callbacks = {
  "eufs_msgs::msg",
  "PathIntegralParams",
  _PathIntegralParams__cdr_serialize,
  _PathIntegralParams__cdr_deserialize,
  _PathIntegralParams__get_serialized_size,
  _PathIntegralParams__max_serialized_size
};

static rosidl_message_type_support_t _PathIntegralParams__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_PathIntegralParams__callbacks,
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
get_message_type_support_handle<eufs_msgs::msg::PathIntegralParams>()
{
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_PathIntegralParams__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, eufs_msgs, msg, PathIntegralParams)() {
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_PathIntegralParams__handle;
}

#ifdef __cplusplus
}
#endif
