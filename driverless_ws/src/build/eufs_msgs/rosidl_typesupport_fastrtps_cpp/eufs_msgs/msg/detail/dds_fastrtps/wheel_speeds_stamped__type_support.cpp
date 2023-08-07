// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/WheelSpeedsStamped.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/wheel_speeds_stamped__rosidl_typesupport_fastrtps_cpp.hpp"
#include "eufs_msgs/msg/detail/wheel_speeds_stamped__struct.hpp"

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
bool cdr_serialize(
  const eufs_msgs::msg::WheelSpeeds &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  eufs_msgs::msg::WheelSpeeds &);
size_t get_serialized_size(
  const eufs_msgs::msg::WheelSpeeds &,
  size_t current_alignment);
size_t
max_serialized_size_WheelSpeeds(
  bool & full_bounded,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace eufs_msgs


namespace eufs_msgs
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
cdr_serialize(
  const eufs_msgs::msg::WheelSpeedsStamped & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: header
  std_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.header,
    cdr);
  // Member: speeds
  eufs_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.speeds,
    cdr);
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  eufs_msgs::msg::WheelSpeedsStamped & ros_message)
{
  // Member: header
  std_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.header);

  // Member: speeds
  eufs_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.speeds);

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
get_serialized_size(
  const eufs_msgs::msg::WheelSpeedsStamped & ros_message,
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
  // Member: speeds

  current_alignment +=
    eufs_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.speeds, current_alignment);

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
max_serialized_size_WheelSpeedsStamped(
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

  // Member: speeds
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        eufs_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_WheelSpeeds(
        full_bounded, current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static bool _WheelSpeedsStamped__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::WheelSpeedsStamped *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _WheelSpeedsStamped__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<eufs_msgs::msg::WheelSpeedsStamped *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _WheelSpeedsStamped__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::WheelSpeedsStamped *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _WheelSpeedsStamped__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_WheelSpeedsStamped(full_bounded, 0);
}

static message_type_support_callbacks_t _WheelSpeedsStamped__callbacks = {
  "eufs_msgs::msg",
  "WheelSpeedsStamped",
  _WheelSpeedsStamped__cdr_serialize,
  _WheelSpeedsStamped__cdr_deserialize,
  _WheelSpeedsStamped__get_serialized_size,
  _WheelSpeedsStamped__max_serialized_size
};

static rosidl_message_type_support_t _WheelSpeedsStamped__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_WheelSpeedsStamped__callbacks,
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
get_message_type_support_handle<eufs_msgs::msg::WheelSpeedsStamped>()
{
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_WheelSpeedsStamped__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, eufs_msgs, msg, WheelSpeedsStamped)() {
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_WheelSpeedsStamped__handle;
}

#ifdef __cplusplus
}
#endif
