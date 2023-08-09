// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/ConeWithCovariance.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CONE_WITH_COVARIANCE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__CONE_WITH_COVARIANCE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/cone_with_covariance__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_ConeWithCovariance_covariance
{
public:
  explicit Init_ConeWithCovariance_covariance(::eufs_msgs::msg::ConeWithCovariance & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::ConeWithCovariance covariance(::eufs_msgs::msg::ConeWithCovariance::_covariance_type arg)
  {
    msg_.covariance = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::ConeWithCovariance msg_;
};

class Init_ConeWithCovariance_point
{
public:
  Init_ConeWithCovariance_point()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ConeWithCovariance_covariance point(::eufs_msgs::msg::ConeWithCovariance::_point_type arg)
  {
    msg_.point = std::move(arg);
    return Init_ConeWithCovariance_covariance(msg_);
  }

private:
  ::eufs_msgs::msg::ConeWithCovariance msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::ConeWithCovariance>()
{
  return eufs_msgs::msg::builder::Init_ConeWithCovariance_point();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__CONE_WITH_COVARIANCE__BUILDER_HPP_
