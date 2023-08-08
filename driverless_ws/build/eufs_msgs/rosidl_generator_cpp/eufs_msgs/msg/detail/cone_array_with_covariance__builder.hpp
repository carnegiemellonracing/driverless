// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/ConeArrayWithCovariance.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/cone_array_with_covariance__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_ConeArrayWithCovariance_unknown_color_cones
{
public:
  explicit Init_ConeArrayWithCovariance_unknown_color_cones(::eufs_msgs::msg::ConeArrayWithCovariance & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::ConeArrayWithCovariance unknown_color_cones(::eufs_msgs::msg::ConeArrayWithCovariance::_unknown_color_cones_type arg)
  {
    msg_.unknown_color_cones = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::ConeArrayWithCovariance msg_;
};

class Init_ConeArrayWithCovariance_big_orange_cones
{
public:
  explicit Init_ConeArrayWithCovariance_big_orange_cones(::eufs_msgs::msg::ConeArrayWithCovariance & msg)
  : msg_(msg)
  {}
  Init_ConeArrayWithCovariance_unknown_color_cones big_orange_cones(::eufs_msgs::msg::ConeArrayWithCovariance::_big_orange_cones_type arg)
  {
    msg_.big_orange_cones = std::move(arg);
    return Init_ConeArrayWithCovariance_unknown_color_cones(msg_);
  }

private:
  ::eufs_msgs::msg::ConeArrayWithCovariance msg_;
};

class Init_ConeArrayWithCovariance_orange_cones
{
public:
  explicit Init_ConeArrayWithCovariance_orange_cones(::eufs_msgs::msg::ConeArrayWithCovariance & msg)
  : msg_(msg)
  {}
  Init_ConeArrayWithCovariance_big_orange_cones orange_cones(::eufs_msgs::msg::ConeArrayWithCovariance::_orange_cones_type arg)
  {
    msg_.orange_cones = std::move(arg);
    return Init_ConeArrayWithCovariance_big_orange_cones(msg_);
  }

private:
  ::eufs_msgs::msg::ConeArrayWithCovariance msg_;
};

class Init_ConeArrayWithCovariance_yellow_cones
{
public:
  explicit Init_ConeArrayWithCovariance_yellow_cones(::eufs_msgs::msg::ConeArrayWithCovariance & msg)
  : msg_(msg)
  {}
  Init_ConeArrayWithCovariance_orange_cones yellow_cones(::eufs_msgs::msg::ConeArrayWithCovariance::_yellow_cones_type arg)
  {
    msg_.yellow_cones = std::move(arg);
    return Init_ConeArrayWithCovariance_orange_cones(msg_);
  }

private:
  ::eufs_msgs::msg::ConeArrayWithCovariance msg_;
};

class Init_ConeArrayWithCovariance_blue_cones
{
public:
  explicit Init_ConeArrayWithCovariance_blue_cones(::eufs_msgs::msg::ConeArrayWithCovariance & msg)
  : msg_(msg)
  {}
  Init_ConeArrayWithCovariance_yellow_cones blue_cones(::eufs_msgs::msg::ConeArrayWithCovariance::_blue_cones_type arg)
  {
    msg_.blue_cones = std::move(arg);
    return Init_ConeArrayWithCovariance_yellow_cones(msg_);
  }

private:
  ::eufs_msgs::msg::ConeArrayWithCovariance msg_;
};

class Init_ConeArrayWithCovariance_header
{
public:
  Init_ConeArrayWithCovariance_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ConeArrayWithCovariance_blue_cones header(::eufs_msgs::msg::ConeArrayWithCovariance::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_ConeArrayWithCovariance_blue_cones(msg_);
  }

private:
  ::eufs_msgs::msg::ConeArrayWithCovariance msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::ConeArrayWithCovariance>()
{
  return eufs_msgs::msg::builder::Init_ConeArrayWithCovariance_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__CONE_ARRAY_WITH_COVARIANCE__BUILDER_HPP_
