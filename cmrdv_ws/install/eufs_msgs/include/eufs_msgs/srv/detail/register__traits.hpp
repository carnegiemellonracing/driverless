// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:srv/Register.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__SRV__DETAIL__REGISTER__TRAITS_HPP_
#define EUFS_MSGS__SRV__DETAIL__REGISTER__TRAITS_HPP_

#include "eufs_msgs/srv/detail/register__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::srv::Register_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: node_name
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "node_name: ";
    value_to_yaml(msg.node_name, out);
    out << "\n";
  }

  // member: severity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "severity: ";
    value_to_yaml(msg.severity, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::srv::Register_Request & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::srv::Register_Request>()
{
  return "eufs_msgs::srv::Register_Request";
}

template<>
inline const char * name<eufs_msgs::srv::Register_Request>()
{
  return "eufs_msgs/srv/Register_Request";
}

template<>
struct has_fixed_size<eufs_msgs::srv::Register_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<eufs_msgs::srv::Register_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<eufs_msgs::srv::Register_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::srv::Register_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "id: ";
    value_to_yaml(msg.id, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::srv::Register_Response & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::srv::Register_Response>()
{
  return "eufs_msgs::srv::Register_Response";
}

template<>
inline const char * name<eufs_msgs::srv::Register_Response>()
{
  return "eufs_msgs/srv/Register_Response";
}

template<>
struct has_fixed_size<eufs_msgs::srv::Register_Response>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<eufs_msgs::srv::Register_Response>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<eufs_msgs::srv::Register_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<eufs_msgs::srv::Register>()
{
  return "eufs_msgs::srv::Register";
}

template<>
inline const char * name<eufs_msgs::srv::Register>()
{
  return "eufs_msgs/srv/Register";
}

template<>
struct has_fixed_size<eufs_msgs::srv::Register>
  : std::integral_constant<
    bool,
    has_fixed_size<eufs_msgs::srv::Register_Request>::value &&
    has_fixed_size<eufs_msgs::srv::Register_Response>::value
  >
{
};

template<>
struct has_bounded_size<eufs_msgs::srv::Register>
  : std::integral_constant<
    bool,
    has_bounded_size<eufs_msgs::srv::Register_Request>::value &&
    has_bounded_size<eufs_msgs::srv::Register_Response>::value
  >
{
};

template<>
struct is_service<eufs_msgs::srv::Register>
  : std::true_type
{
};

template<>
struct is_service_request<eufs_msgs::srv::Register_Request>
  : std::true_type
{
};

template<>
struct is_service_response<eufs_msgs::srv::Register_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__SRV__DETAIL__REGISTER__TRAITS_HPP_
