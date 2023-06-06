// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:msg/TopicStatus.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__TRAITS_HPP_
#define EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__TRAITS_HPP_

#include "eufs_msgs/msg/detail/topic_status__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::msg::TopicStatus & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: topic
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "topic: ";
    value_to_yaml(msg.topic, out);
    out << "\n";
  }

  // member: description
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "description: ";
    value_to_yaml(msg.description, out);
    out << "\n";
  }

  // member: group
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "group: ";
    value_to_yaml(msg.group, out);
    out << "\n";
  }

  // member: trigger_ebs
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "trigger_ebs: ";
    value_to_yaml(msg.trigger_ebs, out);
    out << "\n";
  }

  // member: log_level
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "log_level: ";
    value_to_yaml(msg.log_level, out);
    out << "\n";
  }

  // member: status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status: ";
    value_to_yaml(msg.status, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::msg::TopicStatus & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::msg::TopicStatus>()
{
  return "eufs_msgs::msg::TopicStatus";
}

template<>
inline const char * name<eufs_msgs::msg::TopicStatus>()
{
  return "eufs_msgs/msg/TopicStatus";
}

template<>
struct has_fixed_size<eufs_msgs::msg::TopicStatus>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<eufs_msgs::msg::TopicStatus>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<eufs_msgs::msg::TopicStatus>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__TRAITS_HPP_
