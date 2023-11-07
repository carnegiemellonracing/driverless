// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from cmrdv_interfaces:msg/Brakes.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "cmrdv_interfaces/msg/detail/brakes__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace cmrdv_interfaces
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void Brakes_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) cmrdv_interfaces::msg::Brakes(_init);
}

void Brakes_fini_function(void * message_memory)
{
  auto typed_message = static_cast<cmrdv_interfaces::msg::Brakes *>(message_memory);
  typed_message->~Brakes();
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Brakes_message_member_array[2] = {
  {
    "braking",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_BOOLEAN,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces::msg::Brakes, braking),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "last_fired",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<builtin_interfaces::msg::Time>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(cmrdv_interfaces::msg::Brakes, last_fired),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Brakes_message_members = {
  "cmrdv_interfaces::msg",  // message namespace
  "Brakes",  // message name
  2,  // number of fields
  sizeof(cmrdv_interfaces::msg::Brakes),
  Brakes_message_member_array,  // message members
  Brakes_init_function,  // function to initialize message memory (memory has to be allocated)
  Brakes_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Brakes_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Brakes_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace cmrdv_interfaces


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<cmrdv_interfaces::msg::Brakes>()
{
  return &::cmrdv_interfaces::msg::rosidl_typesupport_introspection_cpp::Brakes_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, cmrdv_interfaces, msg, Brakes)() {
  return &::cmrdv_interfaces::msg::rosidl_typesupport_introspection_cpp::Brakes_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
