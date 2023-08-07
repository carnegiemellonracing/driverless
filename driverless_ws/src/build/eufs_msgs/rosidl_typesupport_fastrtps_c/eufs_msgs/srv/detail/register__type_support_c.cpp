// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from eufs_msgs:srv/Register.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/srv/detail/register__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/srv/detail/register__struct.h"
#include "eufs_msgs/srv/detail/register__functions.h"
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

#include "rosidl_runtime_c/string.h"  // node_name
#include "rosidl_runtime_c/string_functions.h"  // node_name

// forward declare type support functions


using _Register_Request__ros_msg_type = eufs_msgs__srv__Register_Request;

static bool _Register_Request__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _Register_Request__ros_msg_type * ros_message = static_cast<const _Register_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: node_name
  {
    const rosidl_runtime_c__String * str = &ros_message->node_name;
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

  // Field name: severity
  {
    cdr << ros_message->severity;
  }

  return true;
}

static bool _Register_Request__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _Register_Request__ros_msg_type * ros_message = static_cast<_Register_Request__ros_msg_type *>(untyped_ros_message);
  // Field name: node_name
  {
    std::string tmp;
    cdr >> tmp;
    if (!ros_message->node_name.data) {
      rosidl_runtime_c__String__init(&ros_message->node_name);
    }
    bool succeeded = rosidl_runtime_c__String__assign(
      &ros_message->node_name,
      tmp.c_str());
    if (!succeeded) {
      fprintf(stderr, "failed to assign string into field 'node_name'\n");
      return false;
    }
  }

  // Field name: severity
  {
    cdr >> ros_message->severity;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__srv__Register_Request(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _Register_Request__ros_msg_type * ros_message = static_cast<const _Register_Request__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name node_name
  current_alignment += padding +
    eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
    (ros_message->node_name.size + 1);
  // field.name severity
  {
    size_t item_size = sizeof(ros_message->severity);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _Register_Request__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__srv__Register_Request(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__srv__Register_Request(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: node_name
  {
    size_t array_size = 1;

    full_bounded = false;
    for (size_t index = 0; index < array_size; ++index) {
      current_alignment += padding +
        eprosima::fastcdr::Cdr::alignment(current_alignment, padding) +
        1;
    }
  }
  // member: severity
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  return current_alignment - initial_alignment;
}

static size_t _Register_Request__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__srv__Register_Request(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_Register_Request = {
  "eufs_msgs::srv",
  "Register_Request",
  _Register_Request__cdr_serialize,
  _Register_Request__cdr_deserialize,
  _Register_Request__get_serialized_size,
  _Register_Request__max_serialized_size
};

static rosidl_message_type_support_t _Register_Request__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_Register_Request,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, srv, Register_Request)() {
  return &_Register_Request__type_support;
}

#if defined(__cplusplus)
}
#endif

// already included above
// #include <cassert>
// already included above
// #include <limits>
// already included above
// #include <string>
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
// already included above
// #include "eufs_msgs/srv/detail/register__struct.h"
// already included above
// #include "eufs_msgs/srv/detail/register__functions.h"
// already included above
// #include "fastcdr/Cdr.h"

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


using _Register_Response__ros_msg_type = eufs_msgs__srv__Register_Response;

static bool _Register_Response__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const _Register_Response__ros_msg_type * ros_message = static_cast<const _Register_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: id
  {
    cdr << ros_message->id;
  }

  return true;
}

static bool _Register_Response__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  _Register_Response__ros_msg_type * ros_message = static_cast<_Register_Response__ros_msg_type *>(untyped_ros_message);
  // Field name: id
  {
    cdr >> ros_message->id;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t get_serialized_size_eufs_msgs__srv__Register_Response(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _Register_Response__ros_msg_type * ros_message = static_cast<const _Register_Response__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // field.name id
  {
    size_t item_size = sizeof(ros_message->id);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

static uint32_t _Register_Response__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_eufs_msgs__srv__Register_Response(
      untyped_ros_message, 0));
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_eufs_msgs
size_t max_serialized_size_eufs_msgs__srv__Register_Response(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;

  // member: id
  {
    size_t array_size = 1;

    current_alignment += array_size * sizeof(uint8_t);
  }

  return current_alignment - initial_alignment;
}

static size_t _Register_Response__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_eufs_msgs__srv__Register_Response(
    full_bounded, 0);
}


static message_type_support_callbacks_t __callbacks_Register_Response = {
  "eufs_msgs::srv",
  "Register_Response",
  _Register_Response__cdr_serialize,
  _Register_Response__cdr_deserialize,
  _Register_Response__get_serialized_size,
  _Register_Response__max_serialized_size
};

static rosidl_message_type_support_t _Register_Response__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_Register_Response,
  get_message_typesupport_handle_function,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, srv, Register_Response)() {
  return &_Register_Response__type_support;
}

#if defined(__cplusplus)
}
#endif

#include "rosidl_typesupport_fastrtps_cpp/service_type_support.h"
#include "rosidl_typesupport_cpp/service_type_support.hpp"
// already included above
// #include "rosidl_typesupport_fastrtps_c/identifier.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "eufs_msgs/srv/register.h"

#if defined(__cplusplus)
extern "C"
{
#endif

static service_type_support_callbacks_t Register__callbacks = {
  "eufs_msgs::srv",
  "Register",
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, srv, Register_Request)(),
  ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, srv, Register_Response)(),
};

static rosidl_service_type_support_t Register__handle = {
  rosidl_typesupport_fastrtps_c__identifier,
  &Register__callbacks,
  get_service_typesupport_handle_function,
};

const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, eufs_msgs, srv, Register)() {
  return &Register__handle;
}

#if defined(__cplusplus)
}
#endif
