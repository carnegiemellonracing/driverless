// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:action/CheckForObjects.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__ACTION__DETAIL__CHECK_FOR_OBJECTS__TRAITS_HPP_
#define EUFS_MSGS__ACTION__DETAIL__CHECK_FOR_OBJECTS__TRAITS_HPP_

#include "eufs_msgs/action/detail/check_for_objects__struct.hpp"
#include <stdint.h>
#include <rosidl_runtime_cpp/traits.hpp>
#include <sstream>
#include <string>
#include <type_traits>

// Include directives for member types
// Member 'image'
#include "sensor_msgs/msg/detail/image__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::action::CheckForObjects_Goal & msg,
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

  // member: image
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "image:\n";
    to_yaml(msg.image, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::action::CheckForObjects_Goal & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::action::CheckForObjects_Goal>()
{
  return "eufs_msgs::action::CheckForObjects_Goal";
}

template<>
inline const char * name<eufs_msgs::action::CheckForObjects_Goal>()
{
  return "eufs_msgs/action/CheckForObjects_Goal";
}

template<>
struct has_fixed_size<eufs_msgs::action::CheckForObjects_Goal>
  : std::integral_constant<bool, has_fixed_size<sensor_msgs::msg::Image>::value> {};

template<>
struct has_bounded_size<eufs_msgs::action::CheckForObjects_Goal>
  : std::integral_constant<bool, has_bounded_size<sensor_msgs::msg::Image>::value> {};

template<>
struct is_message<eufs_msgs::action::CheckForObjects_Goal>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'bounding_boxes'
#include "eufs_msgs/msg/detail/bounding_boxes__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::action::CheckForObjects_Result & msg,
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

  // member: bounding_boxes
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "bounding_boxes:\n";
    to_yaml(msg.bounding_boxes, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::action::CheckForObjects_Result & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::action::CheckForObjects_Result>()
{
  return "eufs_msgs::action::CheckForObjects_Result";
}

template<>
inline const char * name<eufs_msgs::action::CheckForObjects_Result>()
{
  return "eufs_msgs/action/CheckForObjects_Result";
}

template<>
struct has_fixed_size<eufs_msgs::action::CheckForObjects_Result>
  : std::integral_constant<bool, has_fixed_size<eufs_msgs::msg::BoundingBoxes>::value> {};

template<>
struct has_bounded_size<eufs_msgs::action::CheckForObjects_Result>
  : std::integral_constant<bool, has_bounded_size<eufs_msgs::msg::BoundingBoxes>::value> {};

template<>
struct is_message<eufs_msgs::action::CheckForObjects_Result>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::action::CheckForObjects_Feedback & msg,
  std::ostream & out, size_t indentation = 0)
{
  (void)msg;
  (void)indentation;
  out << "null\n";
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::action::CheckForObjects_Feedback & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::action::CheckForObjects_Feedback>()
{
  return "eufs_msgs::action::CheckForObjects_Feedback";
}

template<>
inline const char * name<eufs_msgs::action::CheckForObjects_Feedback>()
{
  return "eufs_msgs/action/CheckForObjects_Feedback";
}

template<>
struct has_fixed_size<eufs_msgs::action::CheckForObjects_Feedback>
  : std::integral_constant<bool, true> {};

template<>
struct has_bounded_size<eufs_msgs::action::CheckForObjects_Feedback>
  : std::integral_constant<bool, true> {};

template<>
struct is_message<eufs_msgs::action::CheckForObjects_Feedback>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"
// Member 'goal'
#include "eufs_msgs/action/detail/check_for_objects__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::action::CheckForObjects_SendGoal_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: goal_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal_id:\n";
    to_yaml(msg.goal_id, out, indentation + 2);
  }

  // member: goal
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal:\n";
    to_yaml(msg.goal, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::action::CheckForObjects_SendGoal_Request & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::action::CheckForObjects_SendGoal_Request>()
{
  return "eufs_msgs::action::CheckForObjects_SendGoal_Request";
}

template<>
inline const char * name<eufs_msgs::action::CheckForObjects_SendGoal_Request>()
{
  return "eufs_msgs/action/CheckForObjects_SendGoal_Request";
}

template<>
struct has_fixed_size<eufs_msgs::action::CheckForObjects_SendGoal_Request>
  : std::integral_constant<bool, has_fixed_size<eufs_msgs::action::CheckForObjects_Goal>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<eufs_msgs::action::CheckForObjects_SendGoal_Request>
  : std::integral_constant<bool, has_bounded_size<eufs_msgs::action::CheckForObjects_Goal>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<eufs_msgs::action::CheckForObjects_SendGoal_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::action::CheckForObjects_SendGoal_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: accepted
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "accepted: ";
    value_to_yaml(msg.accepted, out);
    out << "\n";
  }

  // member: stamp
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "stamp:\n";
    to_yaml(msg.stamp, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::action::CheckForObjects_SendGoal_Response & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::action::CheckForObjects_SendGoal_Response>()
{
  return "eufs_msgs::action::CheckForObjects_SendGoal_Response";
}

template<>
inline const char * name<eufs_msgs::action::CheckForObjects_SendGoal_Response>()
{
  return "eufs_msgs/action/CheckForObjects_SendGoal_Response";
}

template<>
struct has_fixed_size<eufs_msgs::action::CheckForObjects_SendGoal_Response>
  : std::integral_constant<bool, has_fixed_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct has_bounded_size<eufs_msgs::action::CheckForObjects_SendGoal_Response>
  : std::integral_constant<bool, has_bounded_size<builtin_interfaces::msg::Time>::value> {};

template<>
struct is_message<eufs_msgs::action::CheckForObjects_SendGoal_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<eufs_msgs::action::CheckForObjects_SendGoal>()
{
  return "eufs_msgs::action::CheckForObjects_SendGoal";
}

template<>
inline const char * name<eufs_msgs::action::CheckForObjects_SendGoal>()
{
  return "eufs_msgs/action/CheckForObjects_SendGoal";
}

template<>
struct has_fixed_size<eufs_msgs::action::CheckForObjects_SendGoal>
  : std::integral_constant<
    bool,
    has_fixed_size<eufs_msgs::action::CheckForObjects_SendGoal_Request>::value &&
    has_fixed_size<eufs_msgs::action::CheckForObjects_SendGoal_Response>::value
  >
{
};

template<>
struct has_bounded_size<eufs_msgs::action::CheckForObjects_SendGoal>
  : std::integral_constant<
    bool,
    has_bounded_size<eufs_msgs::action::CheckForObjects_SendGoal_Request>::value &&
    has_bounded_size<eufs_msgs::action::CheckForObjects_SendGoal_Response>::value
  >
{
};

template<>
struct is_service<eufs_msgs::action::CheckForObjects_SendGoal>
  : std::true_type
{
};

template<>
struct is_service_request<eufs_msgs::action::CheckForObjects_SendGoal_Request>
  : std::true_type
{
};

template<>
struct is_service_response<eufs_msgs::action::CheckForObjects_SendGoal_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::action::CheckForObjects_GetResult_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: goal_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal_id:\n";
    to_yaml(msg.goal_id, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::action::CheckForObjects_GetResult_Request & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::action::CheckForObjects_GetResult_Request>()
{
  return "eufs_msgs::action::CheckForObjects_GetResult_Request";
}

template<>
inline const char * name<eufs_msgs::action::CheckForObjects_GetResult_Request>()
{
  return "eufs_msgs/action/CheckForObjects_GetResult_Request";
}

template<>
struct has_fixed_size<eufs_msgs::action::CheckForObjects_GetResult_Request>
  : std::integral_constant<bool, has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<eufs_msgs::action::CheckForObjects_GetResult_Request>
  : std::integral_constant<bool, has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<eufs_msgs::action::CheckForObjects_GetResult_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'result'
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::action::CheckForObjects_GetResult_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: status
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "status: ";
    value_to_yaml(msg.status, out);
    out << "\n";
  }

  // member: result
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "result:\n";
    to_yaml(msg.result, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::action::CheckForObjects_GetResult_Response & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::action::CheckForObjects_GetResult_Response>()
{
  return "eufs_msgs::action::CheckForObjects_GetResult_Response";
}

template<>
inline const char * name<eufs_msgs::action::CheckForObjects_GetResult_Response>()
{
  return "eufs_msgs/action/CheckForObjects_GetResult_Response";
}

template<>
struct has_fixed_size<eufs_msgs::action::CheckForObjects_GetResult_Response>
  : std::integral_constant<bool, has_fixed_size<eufs_msgs::action::CheckForObjects_Result>::value> {};

template<>
struct has_bounded_size<eufs_msgs::action::CheckForObjects_GetResult_Response>
  : std::integral_constant<bool, has_bounded_size<eufs_msgs::action::CheckForObjects_Result>::value> {};

template<>
struct is_message<eufs_msgs::action::CheckForObjects_GetResult_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<eufs_msgs::action::CheckForObjects_GetResult>()
{
  return "eufs_msgs::action::CheckForObjects_GetResult";
}

template<>
inline const char * name<eufs_msgs::action::CheckForObjects_GetResult>()
{
  return "eufs_msgs/action/CheckForObjects_GetResult";
}

template<>
struct has_fixed_size<eufs_msgs::action::CheckForObjects_GetResult>
  : std::integral_constant<
    bool,
    has_fixed_size<eufs_msgs::action::CheckForObjects_GetResult_Request>::value &&
    has_fixed_size<eufs_msgs::action::CheckForObjects_GetResult_Response>::value
  >
{
};

template<>
struct has_bounded_size<eufs_msgs::action::CheckForObjects_GetResult>
  : std::integral_constant<
    bool,
    has_bounded_size<eufs_msgs::action::CheckForObjects_GetResult_Request>::value &&
    has_bounded_size<eufs_msgs::action::CheckForObjects_GetResult_Response>::value
  >
{
};

template<>
struct is_service<eufs_msgs::action::CheckForObjects_GetResult>
  : std::true_type
{
};

template<>
struct is_service_request<eufs_msgs::action::CheckForObjects_GetResult_Request>
  : std::true_type
{
};

template<>
struct is_service_response<eufs_msgs::action::CheckForObjects_GetResult_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__traits.hpp"
// Member 'feedback'
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__traits.hpp"

namespace rosidl_generator_traits
{

inline void to_yaml(
  const eufs_msgs::action::CheckForObjects_FeedbackMessage & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: goal_id
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "goal_id:\n";
    to_yaml(msg.goal_id, out, indentation + 2);
  }

  // member: feedback
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "feedback:\n";
    to_yaml(msg.feedback, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const eufs_msgs::action::CheckForObjects_FeedbackMessage & msg)
{
  std::ostringstream out;
  to_yaml(msg, out);
  return out.str();
}

template<>
inline const char * data_type<eufs_msgs::action::CheckForObjects_FeedbackMessage>()
{
  return "eufs_msgs::action::CheckForObjects_FeedbackMessage";
}

template<>
inline const char * name<eufs_msgs::action::CheckForObjects_FeedbackMessage>()
{
  return "eufs_msgs/action/CheckForObjects_FeedbackMessage";
}

template<>
struct has_fixed_size<eufs_msgs::action::CheckForObjects_FeedbackMessage>
  : std::integral_constant<bool, has_fixed_size<eufs_msgs::action::CheckForObjects_Feedback>::value && has_fixed_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct has_bounded_size<eufs_msgs::action::CheckForObjects_FeedbackMessage>
  : std::integral_constant<bool, has_bounded_size<eufs_msgs::action::CheckForObjects_Feedback>::value && has_bounded_size<unique_identifier_msgs::msg::UUID>::value> {};

template<>
struct is_message<eufs_msgs::action::CheckForObjects_FeedbackMessage>
  : std::true_type {};

}  // namespace rosidl_generator_traits


namespace rosidl_generator_traits
{

template<>
struct is_action<eufs_msgs::action::CheckForObjects>
  : std::true_type
{
};

template<>
struct is_action_goal<eufs_msgs::action::CheckForObjects_Goal>
  : std::true_type
{
};

template<>
struct is_action_result<eufs_msgs::action::CheckForObjects_Result>
  : std::true_type
{
};

template<>
struct is_action_feedback<eufs_msgs::action::CheckForObjects_Feedback>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits


#endif  // EUFS_MSGS__ACTION__DETAIL__CHECK_FOR_OBJECTS__TRAITS_HPP_
