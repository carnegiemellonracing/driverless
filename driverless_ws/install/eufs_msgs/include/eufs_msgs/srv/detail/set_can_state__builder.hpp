// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:srv/SetCanState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__SRV__DETAIL__SET_CAN_STATE__BUILDER_HPP_
#define EUFS_MSGS__SRV__DETAIL__SET_CAN_STATE__BUILDER_HPP_

#include "eufs_msgs/srv/detail/set_can_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace srv
{

namespace builder
{

class Init_SetCanState_Request_as_state
{
public:
  explicit Init_SetCanState_Request_as_state(::eufs_msgs::srv::SetCanState_Request & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::srv::SetCanState_Request as_state(::eufs_msgs::srv::SetCanState_Request::_as_state_type arg)
  {
    msg_.as_state = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::srv::SetCanState_Request msg_;
};

class Init_SetCanState_Request_ami_state
{
public:
  Init_SetCanState_Request_ami_state()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetCanState_Request_as_state ami_state(::eufs_msgs::srv::SetCanState_Request::_ami_state_type arg)
  {
    msg_.ami_state = std::move(arg);
    return Init_SetCanState_Request_as_state(msg_);
  }

private:
  ::eufs_msgs::srv::SetCanState_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::srv::SetCanState_Request>()
{
  return eufs_msgs::srv::builder::Init_SetCanState_Request_ami_state();
}

}  // namespace eufs_msgs


namespace eufs_msgs
{

namespace srv
{

namespace builder
{

class Init_SetCanState_Response_message
{
public:
  explicit Init_SetCanState_Response_message(::eufs_msgs::srv::SetCanState_Response & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::srv::SetCanState_Response message(::eufs_msgs::srv::SetCanState_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::srv::SetCanState_Response msg_;
};

class Init_SetCanState_Response_success
{
public:
  Init_SetCanState_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SetCanState_Response_message success(::eufs_msgs::srv::SetCanState_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_SetCanState_Response_message(msg_);
  }

private:
  ::eufs_msgs::srv::SetCanState_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::srv::SetCanState_Response>()
{
  return eufs_msgs::srv::builder::Init_SetCanState_Response_success();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__SRV__DETAIL__SET_CAN_STATE__BUILDER_HPP_
