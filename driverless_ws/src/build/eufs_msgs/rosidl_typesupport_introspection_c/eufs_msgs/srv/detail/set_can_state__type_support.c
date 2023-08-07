// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:srv/SetCanState.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/srv/detail/set_can_state__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/srv/detail/set_can_state__functions.h"
#include "eufs_msgs/srv/detail/set_can_state__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__srv__SetCanState_Request__init(message_memory);
}

void SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_fini_function(void * message_memory)
{
  eufs_msgs__srv__SetCanState_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_message_member_array[2] = {
  {
    "ami_state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__srv__SetCanState_Request, ami_state),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "as_state",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__srv__SetCanState_Request, as_state),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_message_members = {
  "eufs_msgs__srv",  // message namespace
  "SetCanState_Request",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__srv__SetCanState_Request),
  SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_message_member_array,  // message members
  SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_message_type_support_handle = {
  0,
  &SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, srv, SetCanState_Request)() {
  if (!SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_message_type_support_handle.typesupport_identifier) {
    SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &SetCanState_Request__rosidl_typesupport_introspection_c__SetCanState_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "eufs_msgs/srv/detail/set_can_state__rosidl_typesupport_introspection_c.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "eufs_msgs/srv/detail/set_can_state__functions.h"
// already included above
// #include "eufs_msgs/srv/detail/set_can_state__struct.h"


// Include directives for member types
// Member `message`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__srv__SetCanState_Response__init(message_memory);
}

void SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_fini_function(void * message_memory)
{
  eufs_msgs__srv__SetCanState_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_message_member_array[2] = {
  {
    "success",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__srv__SetCanState_Response, success),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "message",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__srv__SetCanState_Response, message),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_message_members = {
  "eufs_msgs__srv",  // message namespace
  "SetCanState_Response",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__srv__SetCanState_Response),
  SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_message_member_array,  // message members
  SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_message_type_support_handle = {
  0,
  &SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, srv, SetCanState_Response)() {
  if (!SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_message_type_support_handle.typesupport_identifier) {
    SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &SetCanState_Response__rosidl_typesupport_introspection_c__SetCanState_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "eufs_msgs/srv/detail/set_can_state__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers eufs_msgs__srv__detail__set_can_state__rosidl_typesupport_introspection_c__SetCanState_service_members = {
  "eufs_msgs__srv",  // service namespace
  "SetCanState",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // eufs_msgs__srv__detail__set_can_state__rosidl_typesupport_introspection_c__SetCanState_Request_message_type_support_handle,
  NULL  // response message
  // eufs_msgs__srv__detail__set_can_state__rosidl_typesupport_introspection_c__SetCanState_Response_message_type_support_handle
};

static rosidl_service_type_support_t eufs_msgs__srv__detail__set_can_state__rosidl_typesupport_introspection_c__SetCanState_service_type_support_handle = {
  0,
  &eufs_msgs__srv__detail__set_can_state__rosidl_typesupport_introspection_c__SetCanState_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, srv, SetCanState_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, srv, SetCanState_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, srv, SetCanState)() {
  if (!eufs_msgs__srv__detail__set_can_state__rosidl_typesupport_introspection_c__SetCanState_service_type_support_handle.typesupport_identifier) {
    eufs_msgs__srv__detail__set_can_state__rosidl_typesupport_introspection_c__SetCanState_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)eufs_msgs__srv__detail__set_can_state__rosidl_typesupport_introspection_c__SetCanState_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, srv, SetCanState_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, srv, SetCanState_Response)()->data;
  }

  return &eufs_msgs__srv__detail__set_can_state__rosidl_typesupport_introspection_c__SetCanState_service_type_support_handle;
}
