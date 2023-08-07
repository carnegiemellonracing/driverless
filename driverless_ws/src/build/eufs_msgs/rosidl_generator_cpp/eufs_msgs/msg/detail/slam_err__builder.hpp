// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/SLAMErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__SLAM_ERR__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__SLAM_ERR__BUILDER_HPP_

#include "eufs_msgs/msg/detail/slam_err__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_SLAMErr_map_similarity
{
public:
  explicit Init_SLAMErr_map_similarity(::eufs_msgs::msg::SLAMErr & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::SLAMErr map_similarity(::eufs_msgs::msg::SLAMErr::_map_similarity_type arg)
  {
    msg_.map_similarity = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMErr msg_;
};

class Init_SLAMErr_w_orient_err
{
public:
  explicit Init_SLAMErr_w_orient_err(::eufs_msgs::msg::SLAMErr & msg)
  : msg_(msg)
  {}
  Init_SLAMErr_map_similarity w_orient_err(::eufs_msgs::msg::SLAMErr::_w_orient_err_type arg)
  {
    msg_.w_orient_err = std::move(arg);
    return Init_SLAMErr_map_similarity(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMErr msg_;
};

class Init_SLAMErr_z_orient_err
{
public:
  explicit Init_SLAMErr_z_orient_err(::eufs_msgs::msg::SLAMErr & msg)
  : msg_(msg)
  {}
  Init_SLAMErr_w_orient_err z_orient_err(::eufs_msgs::msg::SLAMErr::_z_orient_err_type arg)
  {
    msg_.z_orient_err = std::move(arg);
    return Init_SLAMErr_w_orient_err(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMErr msg_;
};

class Init_SLAMErr_y_orient_err
{
public:
  explicit Init_SLAMErr_y_orient_err(::eufs_msgs::msg::SLAMErr & msg)
  : msg_(msg)
  {}
  Init_SLAMErr_z_orient_err y_orient_err(::eufs_msgs::msg::SLAMErr::_y_orient_err_type arg)
  {
    msg_.y_orient_err = std::move(arg);
    return Init_SLAMErr_z_orient_err(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMErr msg_;
};

class Init_SLAMErr_x_orient_err
{
public:
  explicit Init_SLAMErr_x_orient_err(::eufs_msgs::msg::SLAMErr & msg)
  : msg_(msg)
  {}
  Init_SLAMErr_y_orient_err x_orient_err(::eufs_msgs::msg::SLAMErr::_x_orient_err_type arg)
  {
    msg_.x_orient_err = std::move(arg);
    return Init_SLAMErr_y_orient_err(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMErr msg_;
};

class Init_SLAMErr_z_err
{
public:
  explicit Init_SLAMErr_z_err(::eufs_msgs::msg::SLAMErr & msg)
  : msg_(msg)
  {}
  Init_SLAMErr_x_orient_err z_err(::eufs_msgs::msg::SLAMErr::_z_err_type arg)
  {
    msg_.z_err = std::move(arg);
    return Init_SLAMErr_x_orient_err(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMErr msg_;
};

class Init_SLAMErr_y_err
{
public:
  explicit Init_SLAMErr_y_err(::eufs_msgs::msg::SLAMErr & msg)
  : msg_(msg)
  {}
  Init_SLAMErr_z_err y_err(::eufs_msgs::msg::SLAMErr::_y_err_type arg)
  {
    msg_.y_err = std::move(arg);
    return Init_SLAMErr_z_err(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMErr msg_;
};

class Init_SLAMErr_x_err
{
public:
  explicit Init_SLAMErr_x_err(::eufs_msgs::msg::SLAMErr & msg)
  : msg_(msg)
  {}
  Init_SLAMErr_y_err x_err(::eufs_msgs::msg::SLAMErr::_x_err_type arg)
  {
    msg_.x_err = std::move(arg);
    return Init_SLAMErr_y_err(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMErr msg_;
};

class Init_SLAMErr_header
{
public:
  Init_SLAMErr_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SLAMErr_x_err header(::eufs_msgs::msg::SLAMErr::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_SLAMErr_x_err(msg_);
  }

private:
  ::eufs_msgs::msg::SLAMErr msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::SLAMErr>()
{
  return eufs_msgs::msg::builder::Init_SLAMErr_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__SLAM_ERR__BUILDER_HPP_
