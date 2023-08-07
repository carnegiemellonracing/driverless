// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from eufs_msgs:action/CheckForObjects.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"
#include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "eufs_msgs/action/detail/check_for_objects__functions.h"
#include "eufs_msgs/action/detail/check_for_objects__struct.h"


// Include directives for member types
// Member `image`
#include "sensor_msgs/msg/image.h"
// Member `image`
#include "sensor_msgs/msg/detail/image__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__action__CheckForObjects_Goal__init(message_memory);
}

void CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_fini_function(void * message_memory)
{
  eufs_msgs__action__CheckForObjects_Goal__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_message_member_array[2] = {
  {
    "id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_Goal, id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "image",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_Goal, image),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_message_members = {
  "eufs_msgs__action",  // message namespace
  "CheckForObjects_Goal",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__action__CheckForObjects_Goal),
  CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_message_member_array,  // message members
  CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_init_function,  // function to initialize message memory (memory has to be allocated)
  CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_message_type_support_handle = {
  0,
  &CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_Goal)() {
  CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, sensor_msgs, msg, Image)();
  if (!CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_message_type_support_handle.typesupport_identifier) {
    CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &CheckForObjects_Goal__rosidl_typesupport_introspection_c__CheckForObjects_Goal_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__functions.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__struct.h"


// Include directives for member types
// Member `bounding_boxes`
#include "eufs_msgs/msg/bounding_boxes.h"
// Member `bounding_boxes`
#include "eufs_msgs/msg/detail/bounding_boxes__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__action__CheckForObjects_Result__init(message_memory);
}

void CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_fini_function(void * message_memory)
{
  eufs_msgs__action__CheckForObjects_Result__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_message_member_array[2] = {
  {
    "id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_Result, id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "bounding_boxes",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_Result, bounding_boxes),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_message_members = {
  "eufs_msgs__action",  // message namespace
  "CheckForObjects_Result",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__action__CheckForObjects_Result),
  CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_message_member_array,  // message members
  CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_init_function,  // function to initialize message memory (memory has to be allocated)
  CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_message_type_support_handle = {
  0,
  &CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_Result)() {
  CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, msg, BoundingBoxes)();
  if (!CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_message_type_support_handle.typesupport_identifier) {
    CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &CheckForObjects_Result__rosidl_typesupport_introspection_c__CheckForObjects_Result_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__functions.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__action__CheckForObjects_Feedback__init(message_memory);
}

void CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_fini_function(void * message_memory)
{
  eufs_msgs__action__CheckForObjects_Feedback__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_message_member_array[1] = {
  {
    "structure_needs_at_least_one_member",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_Feedback, structure_needs_at_least_one_member),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_message_members = {
  "eufs_msgs__action",  // message namespace
  "CheckForObjects_Feedback",  // message name
  1,  // number of fields
  sizeof(eufs_msgs__action__CheckForObjects_Feedback),
  CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_message_member_array,  // message members
  CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_init_function,  // function to initialize message memory (memory has to be allocated)
  CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_message_type_support_handle = {
  0,
  &CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_Feedback)() {
  if (!CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_message_type_support_handle.typesupport_identifier) {
    CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &CheckForObjects_Feedback__rosidl_typesupport_introspection_c__CheckForObjects_Feedback_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__functions.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__struct.h"


// Include directives for member types
// Member `goal_id`
#include "unique_identifier_msgs/msg/uuid.h"
// Member `goal_id`
#include "unique_identifier_msgs/msg/detail/uuid__rosidl_typesupport_introspection_c.h"
// Member `goal`
#include "eufs_msgs/action/check_for_objects.h"
// Member `goal`
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__action__CheckForObjects_SendGoal_Request__init(message_memory);
}

void CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_fini_function(void * message_memory)
{
  eufs_msgs__action__CheckForObjects_SendGoal_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_message_member_array[2] = {
  {
    "goal_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_SendGoal_Request, goal_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "goal",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_SendGoal_Request, goal),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_message_members = {
  "eufs_msgs__action",  // message namespace
  "CheckForObjects_SendGoal_Request",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Request),
  CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_message_member_array,  // message members
  CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_message_type_support_handle = {
  0,
  &CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_SendGoal_Request)() {
  CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, unique_identifier_msgs, msg, UUID)();
  CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_Goal)();
  if (!CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_message_type_support_handle.typesupport_identifier) {
    CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &CheckForObjects_SendGoal_Request__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__functions.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__struct.h"


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/time.h"
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__action__CheckForObjects_SendGoal_Response__init(message_memory);
}

void CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_fini_function(void * message_memory)
{
  eufs_msgs__action__CheckForObjects_SendGoal_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_message_member_array[2] = {
  {
    "accepted",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_SendGoal_Response, accepted),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "stamp",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_SendGoal_Response, stamp),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_message_members = {
  "eufs_msgs__action",  // message namespace
  "CheckForObjects_SendGoal_Response",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Response),
  CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_message_member_array,  // message members
  CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_message_type_support_handle = {
  0,
  &CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_SendGoal_Response)() {
  CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, builtin_interfaces, msg, Time)();
  if (!CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_message_type_support_handle.typesupport_identifier) {
    CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &CheckForObjects_SendGoal_Response__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

#include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_service_members = {
  "eufs_msgs__action",  // service namespace
  "CheckForObjects_SendGoal",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Request_message_type_support_handle,
  NULL  // response message
  // eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_Response_message_type_support_handle
};

static rosidl_service_type_support_t eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_service_type_support_handle = {
  0,
  &eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_SendGoal_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_SendGoal_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_SendGoal)() {
  if (!eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_service_type_support_handle.typesupport_identifier) {
    eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_SendGoal_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_SendGoal_Response)()->data;
  }

  return &eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_SendGoal_service_type_support_handle;
}

// already included above
// #include <stddef.h>
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__functions.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__struct.h"


// Include directives for member types
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/uuid.h"
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__action__CheckForObjects_GetResult_Request__init(message_memory);
}

void CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_fini_function(void * message_memory)
{
  eufs_msgs__action__CheckForObjects_GetResult_Request__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_message_member_array[1] = {
  {
    "goal_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_GetResult_Request, goal_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_message_members = {
  "eufs_msgs__action",  // message namespace
  "CheckForObjects_GetResult_Request",  // message name
  1,  // number of fields
  sizeof(eufs_msgs__action__CheckForObjects_GetResult_Request),
  CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_message_member_array,  // message members
  CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_init_function,  // function to initialize message memory (memory has to be allocated)
  CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_message_type_support_handle = {
  0,
  &CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_GetResult_Request)() {
  CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, unique_identifier_msgs, msg, UUID)();
  if (!CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_message_type_support_handle.typesupport_identifier) {
    CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &CheckForObjects_GetResult_Request__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include <stddef.h>
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__functions.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__struct.h"


// Include directives for member types
// Member `result`
// already included above
// #include "eufs_msgs/action/check_for_objects.h"
// Member `result`
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__action__CheckForObjects_GetResult_Response__init(message_memory);
}

void CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_fini_function(void * message_memory)
{
  eufs_msgs__action__CheckForObjects_GetResult_Response__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_message_member_array[2] = {
  {
    "status",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT8,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_GetResult_Response, status),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "result",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_GetResult_Response, result),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_message_members = {
  "eufs_msgs__action",  // message namespace
  "CheckForObjects_GetResult_Response",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__action__CheckForObjects_GetResult_Response),
  CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_message_member_array,  // message members
  CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_init_function,  // function to initialize message memory (memory has to be allocated)
  CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_message_type_support_handle = {
  0,
  &CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_GetResult_Response)() {
  CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_Result)();
  if (!CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_message_type_support_handle.typesupport_identifier) {
    CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &CheckForObjects_GetResult_Response__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

// already included above
// #include "rosidl_runtime_c/service_type_support_struct.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/service_introspection.h"

// this is intentionally not const to allow initialization later to prevent an initialization race
static rosidl_typesupport_introspection_c__ServiceMembers eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_service_members = {
  "eufs_msgs__action",  // service namespace
  "CheckForObjects_GetResult",  // service name
  // these two fields are initialized below on the first access
  NULL,  // request message
  // eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Request_message_type_support_handle,
  NULL  // response message
  // eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_Response_message_type_support_handle
};

static rosidl_service_type_support_t eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_service_type_support_handle = {
  0,
  &eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_service_members,
  get_service_typesupport_handle_function,
};

// Forward declaration of request/response type support functions
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_GetResult_Request)();

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_GetResult_Response)();

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_service_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__SERVICE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_GetResult)() {
  if (!eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_service_type_support_handle.typesupport_identifier) {
    eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_service_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  rosidl_typesupport_introspection_c__ServiceMembers * service_members =
    (rosidl_typesupport_introspection_c__ServiceMembers *)eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_service_type_support_handle.data;

  if (!service_members->request_members_) {
    service_members->request_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_GetResult_Request)()->data;
  }
  if (!service_members->response_members_) {
    service_members->response_members_ =
      (const rosidl_typesupport_introspection_c__MessageMembers *)
      ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_GetResult_Response)()->data;
  }

  return &eufs_msgs__action__detail__check_for_objects__rosidl_typesupport_introspection_c__CheckForObjects_GetResult_service_type_support_handle;
}

// already included above
// #include <stddef.h>
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"
// already included above
// #include "eufs_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
// already included above
// #include "rosidl_typesupport_introspection_c/field_types.h"
// already included above
// #include "rosidl_typesupport_introspection_c/identifier.h"
// already included above
// #include "rosidl_typesupport_introspection_c/message_introspection.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__functions.h"
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__struct.h"


// Include directives for member types
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/uuid.h"
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__rosidl_typesupport_introspection_c.h"
// Member `feedback`
// already included above
// #include "eufs_msgs/action/check_for_objects.h"
// Member `feedback`
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  eufs_msgs__action__CheckForObjects_FeedbackMessage__init(message_memory);
}

void CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_fini_function(void * message_memory)
{
  eufs_msgs__action__CheckForObjects_FeedbackMessage__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_message_member_array[2] = {
  {
    "goal_id",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_FeedbackMessage, goal_id),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "feedback",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(eufs_msgs__action__CheckForObjects_FeedbackMessage, feedback),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_message_members = {
  "eufs_msgs__action",  // message namespace
  "CheckForObjects_FeedbackMessage",  // message name
  2,  // number of fields
  sizeof(eufs_msgs__action__CheckForObjects_FeedbackMessage),
  CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_message_member_array,  // message members
  CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_init_function,  // function to initialize message memory (memory has to be allocated)
  CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_message_type_support_handle = {
  0,
  &CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_eufs_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_FeedbackMessage)() {
  CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, unique_identifier_msgs, msg, UUID)();
  CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, eufs_msgs, action, CheckForObjects_Feedback)();
  if (!CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_message_type_support_handle.typesupport_identifier) {
    CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &CheckForObjects_FeedbackMessage__rosidl_typesupport_introspection_c__CheckForObjects_FeedbackMessage_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
