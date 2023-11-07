// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from cmrdv_interfaces:msg/DataFrame.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/data_frame__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "cmrdv_interfaces/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "cmrdv_interfaces/msg/detail/data_frame__struct.h"
#include "cmrdv_interfaces/msg/detail/data_frame__functions.h"
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

#include "sbg_driver/msg/detail/sbg_gps_pos__functions.h"  // sbg
#include "sbg_driver/msg/detail/sbg_imu_data__functions.h"  // imu
#include "sensor_msgs/msg/detail/image__functions.h"  // zed_left_img
#include "sensor_msgs/msg/detail/point_cloud2__functions.h"  // zed_pts

// forward declare type support functions
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t get_serialized_size_sbg_driver__msg__SbgGpsPos(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t max_serialized_size_sbg_driver__msg__SbgGpsPos(
  bool & full_bounded,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, sbg_driver, msg, SbgGpsPos)();
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t get_serialized_size_sbg_driver__msg__SbgImuData(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t max_serialized_size_sbg_driver__msg__SbgImuData(
  bool & full_bounded,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, sbg_driver, msg, SbgImuData)();
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t get_serialized_size_sensor_msgs__msg__Image(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t max_serialized_size_sensor_msgs__msg__Image(
  bool & full_bounded,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, sensor_msgs, msg, Image)();
ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t get_serialized_size_sensor_msgs__msg__PointCloud2(
  const void * untyped_ros_message,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
size_t max_serialized_size_sensor_msgs__msg__PointCloud2(
  bool & full_bounded,
  size_t current_alignment);

ROSIDL_TYPESUPPORT_FASTRTPS_C_IMPORT_cmrdv_interfaces
const rosidl_message_type_support_t *
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, sensor_msgs, msg, PointCloud2)();


using _DataFrame__ros_msg_type = cmrdv_interfaces__msg__DataFrame;

static bool _DataFrame__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _DataFrame__ros_msg_type * ros_message = static_cast<const _DataFrame__ros_msg_type *>(untyped_ros_message);
  // Field name: zed_left_img
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, sensor_msgs, msg, Image
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->zed_left_img, cdr))
    {
      return false;
    }
  }

  // Field name: zed_pts
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, sensor_msgs, msg, PointCloud2
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->zed_pts, cdr))
    {
      return false;
    }
  }

  // Field name: sbg
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, sbg_driver, msg, SbgGpsPos
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->sbg, cdr))
    {
      return false;
    }
  }

  // Field name: imu
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, sbg_driver, msg, SbgImuData
      )()->data);
    if (!callbacks->cdr_serialize(
        &ros_message->imu, cdr))
    {
      return false;
    }
  }

  return true;
}

static bool _DataFrame__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _DataFrame__ros_msg_type * ros_message = static_cast<_DataFrame__ros_msg_type *>(untyped_ros_message);
  // Field name: zed_left_img
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, sensor_msgs, msg, Image
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->zed_left_img))
    {
      return false;
    }
  }

  // Field name: zed_pts
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, sensor_msgs, msg, PointCloud2
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->zed_pts))
    {
      return false;
    }
  }

  // Field name: sbg
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, sbg_driver, msg, SbgGpsPos
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->sbg))
    {
      return false;
    }
  }

  // Field name: imu
  {
    const message_type_support_callbacks_t * callbacks =
      static_cast<const message_type_support_callbacks_t *>(
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
        rosidl_typesupport_fastrtps_c, sbg_driver, msg, SbgImuData
      )()->data);
    if (!callbacks->cdr_deserialize(
        cdr, &ros_message->imu))
    {
      return false;
    }
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_cmrdv_interfaces
size_t get_serialized_size_cmrdv_interfaces__msg__DataFrame(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _DataFrame__ros_msg_type * ros_message = static_cast<const _DataFrame__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name zed_left_img

  current_alignment += get_serialized_size_sensor_msgs__msg__Image(
    &(ros_message->zed_left_img), current_alignment);
  // field.name zed_pts

  current_alignment += get_serialized_size_sensor_msgs__msg__PointCloud2(
    &(ros_message->zed_pts), current_alignment);
  // field.name sbg

  current_alignment += get_serialized_size_sbg_driver__msg__SbgGpsPos(
    &(ros_message->sbg), current_alignment);
  // field.name imu

  current_alignment += get_serialized_size_sbg_driver__msg__SbgImuData(
    &(ros_message->imu), current_alignment);

  return current_alignment - initial_alignment;
}

static uint32_t _DataFrame__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_cmrdv_interfaces__msg__DataFrame(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_cmrdv_interfaces
size_t max_serialized_size_cmrdv_interfaces__msg__DataFrame(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: zed_left_img
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_sensor_msgs__msg__Image(
        full_bounded, current_alignment);
    }
  }
  // member: zed_pts
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_sensor_msgs__msg__PointCloud2(
        full_bounded, current_alignment);
    }
  }
  // member: sbg
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_sbg_driver__msg__SbgGpsPos(
        full_bounded, current_alignment);
    }
  }
  // member: imu
  {
    size_t array_size = 1;


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        max_serialized_size_sbg_driver__msg__SbgImuData(
        full_bounded, current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static size_t _DataFrame__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_cmrdv_interfaces__msg__DataFrame(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_DataFrame = {
  "cmrdv_interfaces::msg",
  "DataFrame",
  _DataFrame__cdr_serialize,
  _DataFrame__cdr_deserialize,
  _DataFrame__get_serialized_size,
  _DataFrame__max_serialized_size
};

static rosidl_message_type_support_t _DataFrame__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_DataFrame,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, cmrdv_interfaces, msg, DataFrame)() {
  return &_DataFrame__type_support;
}

#if defined(__cplusplus)
}
#endif
