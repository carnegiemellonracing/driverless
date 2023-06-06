// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:srv/Register.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__SRV__DETAIL__REGISTER__BUILDER_HPP_
#define EUFS_MSGS__SRV__DETAIL__REGISTER__BUILDER_HPP_

#include "eufs_msgs/srv/detail/register__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace srv
{

namespace builder
{

class Init_Register_Request_severity
{
public:
  explicit Init_Register_Request_severity(::eufs_msgs::srv::Register_Request & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::srv::Register_Request severity(::eufs_msgs::srv::Register_Request::_severity_type arg)
  {
    msg_.severity = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::srv::Register_Request msg_;
};

class Init_Register_Request_node_name
{
public:
  Init_Register_Request_node_name()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Register_Request_severity node_name(::eufs_msgs::srv::Register_Request::_node_name_type arg)
  {
    msg_.node_name = std::move(arg);
    return Init_Register_Request_severity(msg_);
  }

private:
  ::eufs_msgs::srv::Register_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::srv::Register_Request>()
{
  return eufs_msgs::srv::builder::Init_Register_Request_node_name();
}

}  // namespace eufs_msgs


namespace eufs_msgs
{

namespace srv
{

namespace builder
{

class Init_Register_Response_id
{
public:
  Init_Register_Response_id()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::eufs_msgs::srv::Register_Response id(::eufs_msgs::srv::Register_Response::_id_type arg)
  {
    msg_.id = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::srv::Register_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::srv::Register_Response>()
{
  return eufs_msgs::srv::builder::Init_Register_Response_id();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__SRV__DETAIL__REGISTER__BUILDER_HPP_
