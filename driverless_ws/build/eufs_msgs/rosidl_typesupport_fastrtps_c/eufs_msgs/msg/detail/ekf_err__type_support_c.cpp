// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:msg/EKFErr.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/ekf_err__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/msg/detail/ekf_err__struct.h"
#include "eufs_msgs/msg/detail/ekf_err__functions.h"
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


using _EKFErr__ros_msg_type = eufs_msgs__msg__EKFErr;

static bool _EKFErr__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _EKFErr__ros_msg_type * ros_message = static_cast<const _EKFErr__ros_msg_type *>(untyped_ros_message);
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

  // Field name: gps_x_vel_err
  {
    cdr << ros_message->gps_x_vel_err;
  }

  // Field name: gps_y_vel_err
  {
    cdr << ros_message->gps_y_vel_err;
  }

  // Field name: imu_x_acc_err
  {
    cdr << ros_message->imu_x_acc_err;
  }

  // Field name: imu_y_acc_err
  {
    cdr << ros_message->imu_y_acc_err;
  }

  // Field name: imu_yaw_err
  {
    cdr << ros_message->imu_yaw_err;
  }

  // Field name: ekf_x_vel_var
  {
    cdr << ros_message->ekf_x_vel_var;
  }

  // Field name: ekf_y_vel_var
  {
    cdr << ros_message->ekf_y_vel_var;
  }

  // Field name: ekf_x_acc_var
  {
    cdr << ros_message->ekf_x_acc_var;
  }

  // Field name: ekf_y_acc_var
  {
    cdr << ros_message->ekf_y_acc_var;
  }

  // Field name: ekf_yaw_var
  {
    cdr << ros_message->ekf_yaw_var;
  }

  return true;
}

static bool _EKFErr__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _EKFErr__ros_msg_type * ros_message = static_cast<_EKFErr__ros_msg_type *>(untyped_ros_message);
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

  // Field name: gps_x_vel_err
  {
    cdr >> ros_message->gps_x_vel_err;
  }

  // Field name: gps_y_vel_err
  {
    cdr >> ros_message->gps_y_vel_err;
  }

  // Field name: imu_x_acc_err
  {
    cdr >> ros_message->imu_x_acc_err;
  }

  // Field name: imu_y_acc_err
  {
    cdr >> ros_message->imu_y_acc_err;
  }

  // Field name: imu_yaw_err
  {
    cdr >> ros_message->imu_yaw_err;
  }

  // Field name: ekf_x_vel_var
  {
    cdr >> ros_message->ekf_x_vel_var;
  }

  // Field name: ekf_y_vel_var
  {
    cdr >> ros_message->ekf_y_vel_var;
  }

  // Field name: ekf_x_acc_var
  {
    cdr >> ros_message->ekf_x_acc_var;
  }

  // Field name: ekf_y_acc_var
  {
    cdr >> ros_message->ekf_y_acc_var;
  }

  // Field name: ekf_yaw_var
  {
    cdr >> ros_message->ekf_yaw_var;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__msg__EKFErr(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _EKFErr__ros_msg_type * ros_message = static_cast<const _EKFErr__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name header

  current_alignment += get_serialized_size_std_msgs__msg__Header(
    &(ros_message->header), current_alignment);
  // field.name gps_x_vel_err
  {
    size_t item_size = sizeof(ros_message->gps_x_vel_err);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name gps_y_vel_err
  {
    size_t item_size = sizeof(ros_message->gps_y_vel_err);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name imu_x_acc_err
  {
    size_t item_size = sizeof(ros_message->imu_x_acc_err);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name imu_y_acc_err
  {
    size_t item_size = sizeof(ros_message->imu_y_acc_err);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name imu_yaw_err
  {
    size_t item_size = sizeof(ros_message->imu_yaw_err);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ekf_x_vel_var
  {
    size_t item_size = sizeof(ros_message->ekf_x_vel_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ekf_y_vel_var
  {
    size_t item_size = sizeof(ros_message->ekf_y_vel_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ekf_x_acc_var
  {
    size_t item_size = sizeof(ros_message->ekf_x_acc_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ekf_y_acc_var
  {
    size_t item_size = sizeof(ros_message->ekf_y_acc_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // field.name ekf_yaw_var
  {
    size_t item_size = sizeof(ros_message->ekf_yaw_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _EKFErr__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__msg__EKFErr(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__msg__EKFErr(
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
  // member: gps_x_vel_err
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: gps_y_vel_err
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: imu_x_acc_err
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: imu_y_acc_err
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: imu_yaw_err
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: ekf_x_vel_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: ekf_y_vel_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: ekf_x_acc_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: ekf_y_acc_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }
  // member: ekf_yaw_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static size_t _EKFErr__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__msg__EKFErr(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_EKFErr = {
  "eufs_msgs::msg",
  "EKFErr",
  _EKFErr__cdr_serialize,
  _EKFErr__cdr_deserialize,
  _EKFErr__get_serialized_size,
  _EKFErr__max_serialized_size
};

static rosidl_message_type_support_t _EKFErr__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_EKFErr,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, msg, EKFErr)() {
  return &_EKFErr__type_support;
}

#if defined(__cplusplus)
}
#endif
