// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/PathIntegralStats.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__BUILDER_HPP_

#include "eufs_msgs/msg/detail/path_integral_stats__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_PathIntegralStats_stats
{
public:
  explicit Init_PathIntegralStats_stats(::eufs_msgs::msg::PathIntegralStats & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::PathIntegralStats stats(::eufs_msgs::msg::PathIntegralStats::_stats_type arg)
  {
    msg_.stats = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralStats msg_;
};

class Init_PathIntegralStats_params
{
public:
  explicit Init_PathIntegralStats_params(::eufs_msgs::msg::PathIntegralStats & msg)
  : msg_(msg)
  {}
  Init_PathIntegralStats_stats params(::eufs_msgs::msg::PathIntegralStats::_params_type arg)
  {
    msg_.params = std::move(arg);
    return Init_PathIntegralStats_stats(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralStats msg_;
};

class Init_PathIntegralStats_tag
{
public:
  explicit Init_PathIntegralStats_tag(::eufs_msgs::msg::PathIntegralStats & msg)
  : msg_(msg)
  {}
  Init_PathIntegralStats_params tag(::eufs_msgs::msg::PathIntegralStats::_tag_type arg)
  {
    msg_.tag = std::move(arg);
    return Init_PathIntegralStats_params(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralStats msg_;
};

class Init_PathIntegralStats_header
{
public:
  Init_PathIntegralStats_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PathIntegralStats_tag header(::eufs_msgs::msg::PathIntegralStats::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_PathIntegralStats_tag(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralStats msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::PathIntegralStats>()
{
  return eufs_msgs::msg::builder::Init_PathIntegralStats_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_STATS__BUILDER_HPP_
