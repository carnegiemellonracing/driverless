// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/EKFState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/ekf_state__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/ekf_state__struct.h"
#include "eufs_msgs/msg/detail/ekf_state__functions.h"
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


// forward declare type support functions


using _EKFState__ros_msg_type = eufs_msgs__msg__EKFState;

static bool _EKFState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _EKFState__ros_msg_type * ros_message = static_cast<const _EKFState__ros_msg_type *>(untyped_ros_message);
  // Field name: gps_received
  {
    cdr << (ros_message->gps_received ? true : false);
  }

  // Field name: imu_received
  {
    cdr << (ros_message->imu_received ? true : false);
  }

  // Field name: wheel_odom_received
  {
    cdr << (ros_message->wheel_odom_received ? true : false);
  }

  // Field name: ekf_odom_received
  {
    cdr << (ros_message->ekf_odom_received ? true : false);
  }

  // Field name: ekf_accel_received
  {
    cdr << (ros_message->ekf_accel_received ? true : false);
  }

  // Field name: currently_over_covariance_limit
  {
    cdr << (ros_message->currently_over_covariance_limit ? true : false);
  }

  // Field name: consecutive_turns_over_covariance_limit
  {
    cdr << (ros_message->consecutive_turns_over_covariance_limit ? true : false);
  }

  // Field name: recommends_failure
  {
    cdr << (ros_message->recommends_failure ? true : false);
  }

  return true;
}

static bool _EKFState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _EKFState__ros_msg_type * ros_message = static_cast<_EKFState__ros_msg_type *>(untyped_ros_message);
  // Field name: gps_received
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->gps_received = tmp ? true : false;
  }

  // Field name: imu_received
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->imu_received = tmp ? true : false;
  }

  // Field name: wheel_odom_received
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->wheel_odom_received = tmp ? true : false;
  }

  // Field name: ekf_odom_received
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->ekf_odom_received = tmp ? true : false;
  }

  // Field name: ekf_accel_received
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->ekf_accel_received = tmp ? true : false;
  }

  // Field name: currently_over_covariance_limit
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->currently_over_covariance_limit = tmp ? true : false;
  }

  // Field name: consecutive_turns_over_covariance_limit
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->consecutive_turns_over_covariance_limit = tmp ? true : false;
  }

  // Field name: recommends_failure
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message->recommends_failure = tmp ? true : false;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__EKFState(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _EKFState__ros_msg_type * ros_message = static_cast<const _EKFState__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name gps_received
  {
    size_t item_size = sizeof(ros_message->gps_received);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name imu_received
  {
    size_t item_size = sizeof(ros_message->imu_received);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name wheel_odom_received
  {
    size_t item_size = sizeof(ros_message->wheel_odom_received);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ekf_odom_received
  {
    size_t item_size = sizeof(ros_message->ekf_odom_received);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ekf_accel_received
  {
    size_t item_size = sizeof(ros_message->ekf_accel_received);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name currently_over_covariance_limit
  {
    size_t item_size = sizeof(ros_message->currently_over_covariance_limit);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name consecutive_turns_over_covariance_limit
  {
    size_t item_size = sizeof(ros_message->consecutive_turns_over_covariance_limit);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name recommends_failure
  {
    size_t item_size = sizeof(ros_message->recommends_failure);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _EKFState__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__EKFState(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__EKFState(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: gps_received
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: imu_received
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: wheel_odom_received
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: ekf_odom_received
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: ekf_accel_received
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: currently_over_covariance_limit
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: consecutive_turns_over_covariance_limit
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }
  // member: recommends_failure
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  return current_alignment - initial_alignment;
}

static size_t _EKFState__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__EKFState(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_EKFState = {
  "eufs_msgs::msg",
  "EKFState",
  _EKFState__cdr_serialize,
  _EKFState__cdr_deserialize,
  _EKFState__get_serialized_size,
  _EKFState__max_serialized_size
};

static rosidl_message_type_support_t _EKFState__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_EKFState,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, EKFState)() {
  return &_EKFState__type_support;
}

#if defined(__cplusplus)
}
#endif
