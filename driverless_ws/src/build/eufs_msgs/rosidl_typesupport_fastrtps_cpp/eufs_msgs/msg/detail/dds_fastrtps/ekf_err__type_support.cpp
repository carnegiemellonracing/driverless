// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/EKFErr.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/ekf_err__rosidl_typesupport_fastrtps_cpp.hpp"
#include "eufs_msgs/msg/detail/ekf_err__struct.hpp"

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
namespace std_msgs
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const std_msgs::msg::Header &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  std_msgs::msg::Header &);
size_t get_serialized_size(
  const std_msgs::msg::Header &,
  size_t current_alignment);
size_t
max_serialized_size_Header(
  bool & full_bounded,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace std_msgs


namespace eufs_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
cdr_serialize(
  const eufs_msgs::msg::EKFErr & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: header
  std_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.header,
    cdr);
  // Member: gps_x_vel_err
  cdr << ros_message.gps_x_vel_err;
  // Member: gps_y_vel_err
  cdr << ros_message.gps_y_vel_err;
  // Member: imu_x_acc_err
  cdr << ros_message.imu_x_acc_err;
  // Member: imu_y_acc_err
  cdr << ros_message.imu_y_acc_err;
  // Member: imu_yaw_err
  cdr << ros_message.imu_yaw_err;
  // Member: ekf_x_vel_var
  cdr << ros_message.ekf_x_vel_var;
  // Member: ekf_y_vel_var
  cdr << ros_message.ekf_y_vel_var;
  // Member: ekf_x_acc_var
  cdr << ros_message.ekf_x_acc_var;
  // Member: ekf_y_acc_var
  cdr << ros_message.ekf_y_acc_var;
  // Member: ekf_yaw_var
  cdr << ros_message.ekf_yaw_var;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  eufs_msgs::msg::EKFErr & ros_message)
{
  // Member: header
  std_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.header);

  // Member: gps_x_vel_err
  cdr >> ros_message.gps_x_vel_err;

  // Member: gps_y_vel_err
  cdr >> ros_message.gps_y_vel_err;

  // Member: imu_x_acc_err
  cdr >> ros_message.imu_x_acc_err;

  // Member: imu_y_acc_err
  cdr >> ros_message.imu_y_acc_err;

  // Member: imu_yaw_err
  cdr >> ros_message.imu_yaw_err;

  // Member: ekf_x_vel_var
  cdr >> ros_message.ekf_x_vel_var;

  // Member: ekf_y_vel_var
  cdr >> ros_message.ekf_y_vel_var;

  // Member: ekf_x_acc_var
  cdr >> ros_message.ekf_x_acc_var;

  // Member: ekf_y_acc_var
  cdr >> ros_message.ekf_y_acc_var;

  // Member: ekf_yaw_var
  cdr >> ros_message.ekf_yaw_var;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
get_serialized_size(
  const eufs_msgs::msg::EKFErr & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: header

  current_alignment +=
    std_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.header, current_alignment);
  // Member: gps_x_vel_err
  {
    size_t item_size = sizeof(ros_message.gps_x_vel_err);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: gps_y_vel_err
  {
    size_t item_size = sizeof(ros_message.gps_y_vel_err);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: imu_x_acc_err
  {
    size_t item_size = sizeof(ros_message.imu_x_acc_err);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: imu_y_acc_err
  {
    size_t item_size = sizeof(ros_message.imu_y_acc_err);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: imu_yaw_err
  {
    size_t item_size = sizeof(ros_message.imu_yaw_err);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ekf_x_vel_var
  {
    size_t item_size = sizeof(ros_message.ekf_x_vel_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ekf_y_vel_var
  {
    size_t item_size = sizeof(ros_message.ekf_y_vel_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ekf_x_acc_var
  {
    size_t item_size = sizeof(ros_message.ekf_x_acc_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ekf_y_acc_var
  {
    size_t item_size = sizeof(ros_message.ekf_y_acc_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ekf_yaw_var
  {
    size_t item_size = sizeof(ros_message.ekf_yaw_var);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
max_serialized_size_EKFErr(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: header
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        std_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_Header(
        full_bounded, current_alignment);
    }
  }

  // Member: gps_x_vel_err
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: gps_y_vel_err
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: imu_x_acc_err
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: imu_y_acc_err
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: imu_yaw_err
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: ekf_x_vel_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: ekf_y_vel_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: ekf_x_acc_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: ekf_y_acc_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: ekf_yaw_var
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static bool _EKFErr__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::EKFErr *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _EKFErr__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<eufs_msgs::msg::EKFErr *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _EKFErr__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::EKFErr *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _EKFErr__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_EKFErr(full_bounded, 0);
}

static message_type_support_callbacks_t _EKFErr__callbacks = {
  "eufs_msgs::msg",
  "EKFErr",
  _EKFErr__cdr_serialize,
  _EKFErr__cdr_deserialize,
  _EKFErr__get_serialized_size,
  _EKFErr__max_serialized_size
};

static rosidl_message_type_support_t _EKFErr__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_EKFErr__callbacks,
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
get_message_type_support_handle<eufs_msgs::msg::EKFErr>()
{
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_EKFErr__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, eufs_msgs, msg, EKFErr)() {
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_EKFErr__handle;
}

#ifdef __cplusplus
}
#endif
