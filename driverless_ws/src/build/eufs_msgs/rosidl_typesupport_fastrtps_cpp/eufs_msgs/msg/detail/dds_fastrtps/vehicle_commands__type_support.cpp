// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from eufs_msgs:msg/VehicleCommands.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/vehicle_commands__rosidl_typesupport_fastrtps_cpp.hpp"
#include "eufs_msgs/msg/detail/vehicle_commands__struct.hpp"

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
  const eufs_msgs::msg::VehicleCommands & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: handshake
  cdr << ros_message.handshake;
  // Member: ebs
  cdr << ros_message.ebs;
  // Member: direction
  cdr << ros_message.direction;
  // Member: mission_status
  cdr << ros_message.mission_status;
  // Member: braking
  cdr << ros_message.braking;
  // Member: torque
  cdr << ros_message.torque;
  // Member: steering
  cdr << ros_message.steering;
  // Member: rpm
  cdr << ros_message.rpm;
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  eufs_msgs::msg::VehicleCommands & ros_message)
{
  // Member: handshake
  cdr >> ros_message.handshake;

  // Member: ebs
  cdr >> ros_message.ebs;

  // Member: direction
  cdr >> ros_message.direction;

  // Member: mission_status
  cdr >> ros_message.mission_status;

  // Member: braking
  cdr >> ros_message.braking;

  // Member: torque
  cdr >> ros_message.torque;

  // Member: steering
  cdr >> ros_message.steering;

  // Member: rpm
  cdr >> ros_message.rpm;

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
get_serialized_size(
  const eufs_msgs::msg::VehicleCommands & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: handshake
  {
    size_t item_size = sizeof(ros_message.handshake);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: ebs
  {
    size_t item_size = sizeof(ros_message.ebs);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: direction
  {
    size_t item_size = sizeof(ros_message.direction);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: mission_status
  {
    size_t item_size = sizeof(ros_message.mission_status);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: braking
  {
    size_t item_size = sizeof(ros_message.braking);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: torque
  {
    size_t item_size = sizeof(ros_message.torque);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: steering
  {
    size_t item_size = sizeof(ros_message.steering);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }
  // Member: rpm
  {
    size_t item_size = sizeof(ros_message.rpm);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_eufs_msgs
max_serialized_size_VehicleCommands(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: handshake
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: ebs
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: direction
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: mission_status
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  // Member: braking
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: torque
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: steering
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Member: rpm
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  return current_alignment - initial_alignment;
}

static bool _VehicleCommands__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::VehicleCommands *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _VehicleCommands__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<eufs_msgs::msg::VehicleCommands *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _VehicleCommands__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const eufs_msgs::msg::VehicleCommands *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _VehicleCommands__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_VehicleCommands(full_bounded, 0);
}

static message_type_support_callbacks_t _VehicleCommands__callbacks = {
  "eufs_msgs::msg",
  "VehicleCommands",
  _VehicleCommands__cdr_serialize,
  _VehicleCommands__cdr_deserialize,
  _VehicleCommands__get_serialized_size,
  _VehicleCommands__max_serialized_size
};

static rosidl_message_type_support_t _VehicleCommands__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_VehicleCommands__callbacks,
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
get_message_type_support_handle<eufs_msgs::msg::VehicleCommands>()
{
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_VehicleCommands__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, eufs_msgs, msg, VehicleCommands)() {
  return &eufs_msgs::msg::typesupport_fastrtps_cpp::_VehicleCommands__handle;
}

#ifdef __cplusplus
}
#endif
