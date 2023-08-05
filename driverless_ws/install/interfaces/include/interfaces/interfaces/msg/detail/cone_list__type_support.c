// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from interfaces:msg/ConeList.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "interfaces/msg/detail/cone_list__rosidl_typesupport_introspection_c.h"
#include "interfaces/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "interfaces/msg/detail/cone_list__functions.h"
#include "interfaces/msg/detail/cone_list__struct.h"


// Include directives for member types
// Member `blue_cones`
// Member `yellow_cones`
// Member `orange_cones`
#include "geometry_msgs/msg/point.h"
// Member `blue_cones`
// Member `yellow_cones`
// Member `orange_cones`
#include "geometry_msgs/msg/detail/point__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  interfaces__msg__ConeList__init(message_memory);
}

void interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_fini_function(void * message_memory)
{
  interfaces__msg__ConeList__fini(message_memory);
}

size_t interfaces__msg__ConeList__rosidl_typesupport_introspection_c__size_function__ConeList__blue_cones(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_const_function__ConeList__blue_cones(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_function__ConeList__blue_cones(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void interfaces__msg__ConeList__rosidl_typesupport_introspection_c__fetch_function__ConeList__blue_cones(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__Point * item =
    ((const geometry_msgs__msg__Point *)
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_const_function__ConeList__blue_cones(untyped_member, index));
  geometry_msgs__msg__Point * value =
    (geometry_msgs__msg__Point *)(untyped_value);
  *value = *item;
}

void interfaces__msg__ConeList__rosidl_typesupport_introspection_c__assign_function__ConeList__blue_cones(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__Point * item =
    ((geometry_msgs__msg__Point *)
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_function__ConeList__blue_cones(untyped_member, index));
  const geometry_msgs__msg__Point * value =
    (const geometry_msgs__msg__Point *)(untyped_value);
  *item = *value;
}

bool interfaces__msg__ConeList__rosidl_typesupport_introspection_c__resize_function__ConeList__blue_cones(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

size_t interfaces__msg__ConeList__rosidl_typesupport_introspection_c__size_function__ConeList__yellow_cones(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_const_function__ConeList__yellow_cones(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_function__ConeList__yellow_cones(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void interfaces__msg__ConeList__rosidl_typesupport_introspection_c__fetch_function__ConeList__yellow_cones(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__Point * item =
    ((const geometry_msgs__msg__Point *)
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_const_function__ConeList__yellow_cones(untyped_member, index));
  geometry_msgs__msg__Point * value =
    (geometry_msgs__msg__Point *)(untyped_value);
  *value = *item;
}

void interfaces__msg__ConeList__rosidl_typesupport_introspection_c__assign_function__ConeList__yellow_cones(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__Point * item =
    ((geometry_msgs__msg__Point *)
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_function__ConeList__yellow_cones(untyped_member, index));
  const geometry_msgs__msg__Point * value =
    (const geometry_msgs__msg__Point *)(untyped_value);
  *item = *value;
}

bool interfaces__msg__ConeList__rosidl_typesupport_introspection_c__resize_function__ConeList__yellow_cones(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

size_t interfaces__msg__ConeList__rosidl_typesupport_introspection_c__size_function__ConeList__orange_cones(
  const void * untyped_member)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return member->size;
}

const void * interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_const_function__ConeList__orange_cones(
  const void * untyped_member, size_t index)
{
  const geometry_msgs__msg__Point__Sequence * member =
    (const geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void * interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_function__ConeList__orange_cones(
  void * untyped_member, size_t index)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  return &member->data[index];
}

void interfaces__msg__ConeList__rosidl_typesupport_introspection_c__fetch_function__ConeList__orange_cones(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const geometry_msgs__msg__Point * item =
    ((const geometry_msgs__msg__Point *)
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_const_function__ConeList__orange_cones(untyped_member, index));
  geometry_msgs__msg__Point * value =
    (geometry_msgs__msg__Point *)(untyped_value);
  *value = *item;
}

void interfaces__msg__ConeList__rosidl_typesupport_introspection_c__assign_function__ConeList__orange_cones(
  void * untyped_member, size_t index, const void * untyped_value)
{
  geometry_msgs__msg__Point * item =
    ((geometry_msgs__msg__Point *)
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_function__ConeList__orange_cones(untyped_member, index));
  const geometry_msgs__msg__Point * value =
    (const geometry_msgs__msg__Point *)(untyped_value);
  *item = *value;
}

bool interfaces__msg__ConeList__rosidl_typesupport_introspection_c__resize_function__ConeList__orange_cones(
  void * untyped_member, size_t size)
{
  geometry_msgs__msg__Point__Sequence * member =
    (geometry_msgs__msg__Point__Sequence *)(untyped_member);
  geometry_msgs__msg__Point__Sequence__fini(member);
  return geometry_msgs__msg__Point__Sequence__init(member, size);
}

static rosidl_typesupport_introspection_c__MessageMember interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_message_member_array[3] = {
  {
    "blue_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(interfaces__msg__ConeList, blue_cones),  // bytes offset in struct
    NULL,  // default value
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__size_function__ConeList__blue_cones,  // size() function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_const_function__ConeList__blue_cones,  // get_const(index) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_function__ConeList__blue_cones,  // get(index) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__fetch_function__ConeList__blue_cones,  // fetch(index, &value) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__assign_function__ConeList__blue_cones,  // assign(index, value) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__resize_function__ConeList__blue_cones  // resize(index) function pointer
  },
  {
    "yellow_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(interfaces__msg__ConeList, yellow_cones),  // bytes offset in struct
    NULL,  // default value
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__size_function__ConeList__yellow_cones,  // size() function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_const_function__ConeList__yellow_cones,  // get_const(index) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_function__ConeList__yellow_cones,  // get(index) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__fetch_function__ConeList__yellow_cones,  // fetch(index, &value) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__assign_function__ConeList__yellow_cones,  // assign(index, value) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__resize_function__ConeList__yellow_cones  // resize(index) function pointer
  },
  {
    "orange_cones",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(interfaces__msg__ConeList, orange_cones),  // bytes offset in struct
    NULL,  // default value
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__size_function__ConeList__orange_cones,  // size() function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_const_function__ConeList__orange_cones,  // get_const(index) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__get_function__ConeList__orange_cones,  // get(index) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__fetch_function__ConeList__orange_cones,  // fetch(index, &value) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__assign_function__ConeList__orange_cones,  // assign(index, value) function pointer
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__resize_function__ConeList__orange_cones  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_message_members = {
  "interfaces__msg",  // message namespace
  "ConeList",  // message name
  3,  // number of fields
  sizeof(interfaces__msg__ConeList),
  interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_message_member_array,  // message members
  interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_init_function,  // function to initialize message memory (memory has to be allocated)
  interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_message_type_support_handle = {
  0,
  &interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_message_members,
  get_message_typesupport_handle_function,
  &interfaces__msg__ConeList__get_type_hash,
  &interfaces__msg__ConeList__get_type_description,
  &interfaces__msg__ConeList__get_type_description_sources,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_interfaces
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, interfaces, msg, ConeList)() {
  interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_message_member_array[2].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Point)();
  if (!interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_message_type_support_handle.typesupport_identifier) {
    interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &interfaces__msg__ConeList__rosidl_typesupport_introspection_c__ConeList_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
