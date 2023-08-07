// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/IntegrationErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__INTEGRATION_ERR__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__INTEGRATION_ERR__BUILDER_HPP_

#include "eufs_msgs/msg/detail/integration_err__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_IntegrationErr_angular_vel_err
{
public:
  explicit Init_IntegrationErr_angular_vel_err(::eufs_msgs::msg::IntegrationErr & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::IntegrationErr angular_vel_err(::eufs_msgs::msg::IntegrationErr::_angular_vel_err_type arg)
  {
    msg_.angular_vel_err = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::IntegrationErr msg_;
};

class Init_IntegrationErr_linear_vel_err
{
public:
  explicit Init_IntegrationErr_linear_vel_err(::eufs_msgs::msg::IntegrationErr & msg)
  : msg_(msg)
  {}
  Init_IntegrationErr_angular_vel_err linear_vel_err(::eufs_msgs::msg::IntegrationErr::_linear_vel_err_type arg)
  {
    msg_.linear_vel_err = std::move(arg);
    return Init_IntegrationErr_angular_vel_err(msg_);
  }

private:
  ::eufs_msgs::msg::IntegrationErr msg_;
};

class Init_IntegrationErr_orientation_err
{
public:
  explicit Init_IntegrationErr_orientation_err(::eufs_msgs::msg::IntegrationErr & msg)
  : msg_(msg)
  {}
  Init_IntegrationErr_linear_vel_err orientation_err(::eufs_msgs::msg::IntegrationErr::_orientation_err_type arg)
  {
    msg_.orientation_err = std::move(arg);
    return Init_IntegrationErr_linear_vel_err(msg_);
  }

private:
  ::eufs_msgs::msg::IntegrationErr msg_;
};

class Init_IntegrationErr_position_err
{
public:
  explicit Init_IntegrationErr_position_err(::eufs_msgs::msg::IntegrationErr & msg)
  : msg_(msg)
  {}
  Init_IntegrationErr_orientation_err position_err(::eufs_msgs::msg::IntegrationErr::_position_err_type arg)
  {
    msg_.position_err = std::move(arg);
    return Init_IntegrationErr_orientation_err(msg_);
  }

private:
  ::eufs_msgs::msg::IntegrationErr msg_;
};

class Init_IntegrationErr_header
{
public:
  Init_IntegrationErr_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_IntegrationErr_position_err header(::eufs_msgs::msg::IntegrationErr::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_IntegrationErr_position_err(msg_);
  }

private:
  ::eufs_msgs::msg::IntegrationErr msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::IntegrationErr>()
{
  return eufs_msgs::msg::builder::Init_IntegrationErr_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__INTEGRATION_ERR__BUILDER_HPP_
