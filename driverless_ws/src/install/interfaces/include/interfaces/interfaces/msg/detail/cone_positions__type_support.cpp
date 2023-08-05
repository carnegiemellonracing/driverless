// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from interfaces:msg/ConePositions.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "interfaces/msg/detail/cone_positions__functions.h"
#include "interfaces/msg/detail/cone_positions__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void ConePositions_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) interfaces::msg::ConePositions(_init);
}

void ConePositions_fini_function(void * message_memory)
{
  auto typed_message = static_cast<interfaces::msg::ConePositions *>(message_memory);
  typed_message->~ConePositions();
}

size_t size_function__ConePositions__cone_positions(const void * untyped_member)
{
  const auto * member = reinterpret_cast<const std::vector<std_msgs::msg::Float32> *>(untyped_member);
  return member->size();
}

const void * get_const_function__ConePositions__cone_positions(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::vector<std_msgs::msg::Float32> *>(untyped_member);
  return &member[index];
}

void * get_function__ConePositions__cone_positions(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::vector<std_msgs::msg::Float32> *>(untyped_member);
  return &member[index];
}

void fetch_function__ConePositions__cone_positions(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const std_msgs::msg::Float32 *>(
    get_const_function__ConePositions__cone_positions(untyped_member, index));
  auto & value = *reinterpret_cast<std_msgs::msg::Float32 *>(untyped_value);
  value = item;
}

void assign_function__ConePositions__cone_positions(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<std_msgs::msg::Float32 *>(
    get_function__ConePositions__cone_positions(untyped_member, index));
  const auto & value = *reinterpret_cast<const std_msgs::msg::Float32 *>(untyped_value);
  item = value;
}

void resize_function__ConePositions__cone_positions(void * untyped_member, size_t size)
{
  auto * member =
    reinterpret_cast<std::vector<std_msgs::msg::Float32> *>(untyped_member);
  member->resize(size);
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember ConePositions_message_member_array[2] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(interfaces::msg::ConePositions, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "cone_positions",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Float32>(),  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(interfaces::msg::ConePositions, cone_positions),  // bytes offset in struct
    nullptr,  // default value
    size_function__ConePositions__cone_positions,  // size() function pointer
    get_const_function__ConePositions__cone_positions,  // get_const(index) function pointer
    get_function__ConePositions__cone_positions,  // get(index) function pointer
    fetch_function__ConePositions__cone_positions,  // fetch(index, &value) function pointer
    assign_function__ConePositions__cone_positions,  // assign(index, value) function pointer
    resize_function__ConePositions__cone_positions  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers ConePositions_message_members = {
  "interfaces::msg",  // message namespace
  "ConePositions",  // message name
  2,  // number of fields
  sizeof(interfaces::msg::ConePositions),
  ConePositions_message_member_array,  // message members
  ConePositions_init_function,  // function to initialize message memory (memory has to be allocated)
  ConePositions_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t ConePositions_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &ConePositions_message_members,
  get_message_typesupport_handle_function,
  &interfaces__msg__ConePositions__get_type_hash,
  &interfaces__msg__ConePositions__get_type_description,
  &interfaces__msg__ConePositions__get_type_description_sources,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<interfaces::msg::ConePositions>()
{
  return &::interfaces::msg::rosidl_typesupport_introspection_cpp::ConePositions_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, interfaces, msg, ConePositions)() {
  return &::interfaces::msg::rosidl_typesupport_introspection_cpp::ConePositions_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
