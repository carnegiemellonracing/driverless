// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:action/CheckForObjects.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__ACTION__DETAIL__CHECK_FOR_OBJECTS__BUILDER_HPP_
#define EUFS_MSGS__ACTION__DETAIL__CHECK_FOR_OBJECTS__BUILDER_HPP_

#include "eufs_msgs/action/detail/check_for_objects__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace action
{

namespace builder
{

class Init_CheckForObjects_Goal_image
{
public:
  explicit Init_CheckForObjects_Goal_image(::eufs_msgs::action::CheckForObjects_Goal & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::action::CheckForObjects_Goal image(::eufs_msgs::action::CheckForObjects_Goal::_image_type arg)
  {
    msg_.image = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_Goal msg_;
};

class Init_CheckForObjects_Goal_id
{
public:
  Init_CheckForObjects_Goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CheckForObjects_Goal_image id(::eufs_msgs::action::CheckForObjects_Goal::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_CheckForObjects_Goal_image(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_Goal msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::action::CheckForObjects_Goal>()
{
  return eufs_msgs::action::builder::Init_CheckForObjects_Goal_id();
}

}  // namespace eufs_msgs


namespace eufs_msgs
{

namespace action
{

namespace builder
{

class Init_CheckForObjects_Result_bounding_boxes
{
public:
  explicit Init_CheckForObjects_Result_bounding_boxes(::eufs_msgs::action::CheckForObjects_Result & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::action::CheckForObjects_Result bounding_boxes(::eufs_msgs::action::CheckForObjects_Result::_bounding_boxes_type arg)
  {
    msg_.bounding_boxes = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_Result msg_;
};

class Init_CheckForObjects_Result_id
{
public:
  Init_CheckForObjects_Result_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CheckForObjects_Result_bounding_boxes id(::eufs_msgs::action::CheckForObjects_Result::_id_type arg)
  {
    msg_.id = std::move(arg);
    return Init_CheckForObjects_Result_bounding_boxes(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_Result msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::action::CheckForObjects_Result>()
{
  return eufs_msgs::action::builder::Init_CheckForObjects_Result_id();
}

}  // namespace eufs_msgs


namespace eufs_msgs
{

namespace action
{


}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::action::CheckForObjects_Feedback>()
{
  return ::eufs_msgs::action::CheckForObjects_Feedback(rosidl_runtime_cpp::MessageInitialization::ZERO);
}

}  // namespace eufs_msgs


namespace eufs_msgs
{

namespace action
{

namespace builder
{

class Init_CheckForObjects_SendGoal_Request_goal
{
public:
  explicit Init_CheckForObjects_SendGoal_Request_goal(::eufs_msgs::action::CheckForObjects_SendGoal_Request & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::action::CheckForObjects_SendGoal_Request goal(::eufs_msgs::action::CheckForObjects_SendGoal_Request::_goal_type arg)
  {
    msg_.goal = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_SendGoal_Request msg_;
};

class Init_CheckForObjects_SendGoal_Request_goal_id
{
public:
  Init_CheckForObjects_SendGoal_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CheckForObjects_SendGoal_Request_goal goal_id(::eufs_msgs::action::CheckForObjects_SendGoal_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_CheckForObjects_SendGoal_Request_goal(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_SendGoal_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::action::CheckForObjects_SendGoal_Request>()
{
  return eufs_msgs::action::builder::Init_CheckForObjects_SendGoal_Request_goal_id();
}

}  // namespace eufs_msgs


namespace eufs_msgs
{

namespace action
{

namespace builder
{

class Init_CheckForObjects_SendGoal_Response_stamp
{
public:
  explicit Init_CheckForObjects_SendGoal_Response_stamp(::eufs_msgs::action::CheckForObjects_SendGoal_Response & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::action::CheckForObjects_SendGoal_Response stamp(::eufs_msgs::action::CheckForObjects_SendGoal_Response::_stamp_type arg)
  {
    msg_.stamp = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_SendGoal_Response msg_;
};

class Init_CheckForObjects_SendGoal_Response_accepted
{
public:
  Init_CheckForObjects_SendGoal_Response_accepted()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CheckForObjects_SendGoal_Response_stamp accepted(::eufs_msgs::action::CheckForObjects_SendGoal_Response::_accepted_type arg)
  {
    msg_.accepted = std::move(arg);
    return Init_CheckForObjects_SendGoal_Response_stamp(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_SendGoal_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::action::CheckForObjects_SendGoal_Response>()
{
  return eufs_msgs::action::builder::Init_CheckForObjects_SendGoal_Response_accepted();
}

}  // namespace eufs_msgs


namespace eufs_msgs
{

namespace action
{

namespace builder
{

class Init_CheckForObjects_GetResult_Request_goal_id
{
public:
  Init_CheckForObjects_GetResult_Request_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::eufs_msgs::action::CheckForObjects_GetResult_Request goal_id(::eufs_msgs::action::CheckForObjects_GetResult_Request::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_GetResult_Request msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::action::CheckForObjects_GetResult_Request>()
{
  return eufs_msgs::action::builder::Init_CheckForObjects_GetResult_Request_goal_id();
}

}  // namespace eufs_msgs


namespace eufs_msgs
{

namespace action
{

namespace builder
{

class Init_CheckForObjects_GetResult_Response_result
{
public:
  explicit Init_CheckForObjects_GetResult_Response_result(::eufs_msgs::action::CheckForObjects_GetResult_Response & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::action::CheckForObjects_GetResult_Response result(::eufs_msgs::action::CheckForObjects_GetResult_Response::_result_type arg)
  {
    msg_.result = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_GetResult_Response msg_;
};

class Init_CheckForObjects_GetResult_Response_status
{
public:
  Init_CheckForObjects_GetResult_Response_status()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CheckForObjects_GetResult_Response_result status(::eufs_msgs::action::CheckForObjects_GetResult_Response::_status_type arg)
  {
    msg_.status = std::move(arg);
    return Init_CheckForObjects_GetResult_Response_result(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_GetResult_Response msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::action::CheckForObjects_GetResult_Response>()
{
  return eufs_msgs::action::builder::Init_CheckForObjects_GetResult_Response_status();
}

}  // namespace eufs_msgs


namespace eufs_msgs
{

namespace action
{

namespace builder
{

class Init_CheckForObjects_FeedbackMessage_feedback
{
public:
  explicit Init_CheckForObjects_FeedbackMessage_feedback(::eufs_msgs::action::CheckForObjects_FeedbackMessage & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::action::CheckForObjects_FeedbackMessage feedback(::eufs_msgs::action::CheckForObjects_FeedbackMessage::_feedback_type arg)
  {
    msg_.feedback = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_FeedbackMessage msg_;
};

class Init_CheckForObjects_FeedbackMessage_goal_id
{
public:
  Init_CheckForObjects_FeedbackMessage_goal_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CheckForObjects_FeedbackMessage_feedback goal_id(::eufs_msgs::action::CheckForObjects_FeedbackMessage::_goal_id_type arg)
  {
    msg_.goal_id = std::move(arg);
    return Init_CheckForObjects_FeedbackMessage_feedback(msg_);
  }

private:
  ::eufs_msgs::action::CheckForObjects_FeedbackMessage msg_;
};

}  // namespace builder

}  // namespace action

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::action::CheckForObjects_FeedbackMessage>()
{
  return eufs_msgs::action::builder::Init_CheckForObjects_FeedbackMessage_goal_id();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__ACTION__DETAIL__CHECK_FOR_OBJECTS__BUILDER_HPP_
