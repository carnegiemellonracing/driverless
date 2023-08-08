// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from eufs_msgs:action/CheckForObjects.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__ACTION__DETAIL__CHECK_FOR_OBJECTS__TRAITS_HPP_
#define EUFS_MSGS__ACTION__DETAIL__CHECK_FOR_OBJECTS__TRAITS_HPP_

#include "eufs_msgs/action/detail/check_for_objects__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'image'
#include "sensor_msgs/msg/detail/image__traits.hpp"

namespace rosidl_generator_traits
{

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
