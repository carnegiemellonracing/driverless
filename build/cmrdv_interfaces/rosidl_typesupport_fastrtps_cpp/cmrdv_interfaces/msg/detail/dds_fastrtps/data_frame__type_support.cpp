// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from cmrdv_interfaces:msg/DataFrame.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/data_frame__rosidl_typesupport_fastrtps_cpp.hpp"
#include "cmrdv_interfaces/msg/detail/data_frame__struct.hpp"

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
namespace sensor_msgs
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const sensor_msgs::msg::Image &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  sensor_msgs::msg::Image &);
size_t get_serialized_size(
  const sensor_msgs::msg::Image &,
  size_t current_alignment);
size_t
max_serialized_size_Image(
  bool & full_bounded,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace sensor_msgs

namespace sensor_msgs
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const sensor_msgs::msg::PointCloud2 &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  sensor_msgs::msg::PointCloud2 &);
size_t get_serialized_size(
  const sensor_msgs::msg::PointCloud2 &,
  size_t current_alignment);
size_t
max_serialized_size_PointCloud2(
  bool & full_bounded,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace sensor_msgs

namespace sbg_driver
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const sbg_driver::msg::SbgGpsPos &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  sbg_driver::msg::SbgGpsPos &);
size_t get_serialized_size(
  const sbg_driver::msg::SbgGpsPos &,
  size_t current_alignment);
size_t
max_serialized_size_SbgGpsPos(
  bool & full_bounded,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace sbg_driver

namespace sbg_driver
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const sbg_driver::msg::SbgImuData &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  sbg_driver::msg::SbgImuData &);
size_t get_serialized_size(
  const sbg_driver::msg::SbgImuData &,
  size_t current_alignment);
size_t
max_serialized_size_SbgImuData(
  bool & full_bounded,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace sbg_driver


namespace cmrdv_interfaces
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
cdr_serialize(
  const cmrdv_interfaces::msg::DataFrame & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: zed_left_img
  sensor_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.zed_left_img,
    cdr);
  // Member: zed_pts
  sensor_msgs::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.zed_pts,
    cdr);
  // Member: sbg
  sbg_driver::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.sbg,
    cdr);
  // Member: imu
  sbg_driver::msg::typesupport_fastrtps_cpp::cdr_serialize(
    ros_message.imu,
    cdr);
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  cmrdv_interfaces::msg::DataFrame & ros_message)
{
  // Member: zed_left_img
  sensor_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.zed_left_img);

  // Member: zed_pts
  sensor_msgs::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.zed_pts);

  // Member: sbg
  sbg_driver::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.sbg);

  // Member: imu
  sbg_driver::msg::typesupport_fastrtps_cpp::cdr_deserialize(
    cdr, ros_message.imu);

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
get_serialized_size(
  const cmrdv_interfaces::msg::DataFrame & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: zed_left_img

  current_alignment +=
    sensor_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.zed_left_img, current_alignment);
  // Member: zed_pts

  current_alignment +=
    sensor_msgs::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.zed_pts, current_alignment);
  // Member: sbg

  current_alignment +=
    sbg_driver::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.sbg, current_alignment);
  // Member: imu

  current_alignment +=
    sbg_driver::msg::typesupport_fastrtps_cpp::get_serialized_size(
    ros_message.imu, current_alignment);

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_cmrdv_interfaces
max_serialized_size_DataFrame(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: zed_left_img
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        sensor_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_Image(
        full_bounded, current_alignment);
    }
  }

  // Member: zed_pts
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        sensor_msgs::msg::typesupport_fastrtps_cpp::max_serialized_size_PointCloud2(
        full_bounded, current_alignment);
    }
  }

  // Member: sbg
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        sbg_driver::msg::typesupport_fastrtps_cpp::max_serialized_size_SbgGpsPos(
        full_bounded, current_alignment);
    }
  }

  // Member: imu
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        sbg_driver::msg::typesupport_fastrtps_cpp::max_serialized_size_SbgImuData(
        full_bounded, current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static bool _DataFrame__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const cmrdv_interfaces::msg::DataFrame *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _DataFrame__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<cmrdv_interfaces::msg::DataFrame *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _DataFrame__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const cmrdv_interfaces::msg::DataFrame *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _DataFrame__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_DataFrame(full_bounded, 0);
}

static message_type_support_callbacks_t _DataFrame__callbacks = {
  "cmrdv_interfaces::msg",
  "DataFrame",
  _DataFrame__cdr_serialize,
  _DataFrame__cdr_deserialize,
  _DataFrame__get_serialized_size,
  _DataFrame__max_serialized_size
};

static rosidl_message_type_support_t _DataFrame__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_DataFrame__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace cmrdv_interfaces

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
get_message_type_support_handle<cmrdv_interfaces::msg::DataFrame>()
{
  return &cmrdv_interfaces::msg::typesupport_fastrtps_cpp::_DataFrame__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, cmrdv_interfaces, msg, DataFrame)() {
  return &cmrdv_interfaces::msg::typesupport_fastrtps_cpp::_DataFrame__handle;
}

#ifdef __cplusplus
}
#endif
