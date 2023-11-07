// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/EKFState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/ekf_state__rosidl_typesupport_fastrtps_cpp.hpp"
#include "eufs_msgs/msg/detail/ekf_state__struct.hpp"

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
  const eufs_msgs::msg::EKFState & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: gps_received
  cdr << (ros_message.gps_received ? true : false);
  // Member: imu_received
  cdr << (ros_message.imu_received ? true : false);
  // Member: wheel_odom_received
  cdr << (ros_message.wheel_odom_received ? true : false);
  // Member: ekf_odom_received
  cdr << (ros_message.ekf_odom_received ? true : false);
  // Member: ekf_accel_received
  cdr << (ros_message.ekf_accel_received ? true : false);
  // Member: currently_over_covariance_limit
  cdr << (ros_message.currently_over_covariance_limit ? true : false);
  // Member: consecutive_turns_over_covariance_limit
  cdr << (ros_message.consecutive_turns_over_covariance_limit ? true : false);
  // Member: recommends_failure
  cdr << (ros_message.recommends_failure ? true : false);
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  eufs_msgs::msg::EKFState & ros_message)
{
  // Member: gps_received
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.gps_received = tmp ? true : false;
  }

  // Member: imu_received
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.imu_received = tmp ? true : false;
  }

  // Member: wheel_odom_received
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.wheel_odom_received = tmp ? true : false;
  }

  // Member: ekf_odom_received
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.ekf_odom_received = tmp ? true : false;
  }

  // Member: ekf_accel_received
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.ekf_accel_received = tmp ? true : false;
  }

  // Member: currently_over_covariance_limit
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.currently_over_covariance_limit = tmp ? true : false;
  }

  // Member: consecutive_turns_over_covariance_limit
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.consecutive_turns_over_covariance_limit = tmp ? true : false;
  }

  // Member: recommends_failure
  {
    uint8_t tmp;
    cdr >> tmp;
    ros_message.recommends_failure = tmp ? true : false;
  }

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
get_serialized_size(
  const eufs_msgs::msg::EKFState & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: gps_received
  {
    size_t item_size = sizeof(ros_message.gps_received);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: imu_received
  {
    size_t item_size = sizeof(ros_message.imu_received);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: wheel_odom_received
  {
    size_t item_size = sizeof(ros_message.wheel_odom_received);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ekf_odom_received
  {
    size_t item_size = sizeof(ros_message.ekf_odom_received);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ekf_accel_received
  {
    size_t item_size = sizeof(ros_message.ekf_accel_received);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: currently_over_covariance_limit
  {
    size_t item_size = sizeof(ros_message.currently_over_covariance_limit);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: consecutive_turns_over_covariance_limit
  {
    size_t item_size = sizeof(ros_message.consecutive_turns_over_covariance_limit);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: recommends_failure
  {
    size_t item_size = sizeof(ros_message.recommends_failure);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
max_serialized_size_EKFState(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: gps_received
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: imu_received
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: wheel_odom_received
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: ekf_odom_received
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: ekf_accel_received
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: currently_over_covariance_limit
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: consecutive_turns_over_covariance_limit
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: recommends_failure
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  return current_alignment - initial_alignment;
}

static bool _EKFState__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::EKFState *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _EKFState__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<eufs_msgs::msg::EKFState *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _EKFState__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::EKFState *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _EKFState__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_EKFState(full_bounded, 0);
}

static message_type_support_callbacks_t _EKFState__callbacks = {
  "eufs_msgs::msg",
  "EKFState",
  _EKFState__cdr_serialize,
  _EKFState__cdr_deserialize,
  _EKFState__get_serialized_size,
  _EKFState__max_serialized_size
};

static rosidl_message_type_support_t _EKFState__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_EKFState__callbacks,
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
get_message_type_support_handle<eufs_msgs::msg::EKFState>()
{
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_EKFState__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, eufs_msgs, msg, EKFState)() {
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_EKFState__handle;
}

#ifdef __cplusplus
}
#endif
