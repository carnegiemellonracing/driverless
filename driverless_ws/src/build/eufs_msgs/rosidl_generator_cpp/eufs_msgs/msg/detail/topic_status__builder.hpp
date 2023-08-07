// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/TopicStatus.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__BUILDER_HPP_

#include "eufs_msgs/msg/detail/topic_status__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_TopicStatus_status
{
public:
  explicit Init_TopicStatus_status(::eufs_msgs::msg::TopicStatus & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::TopicStatus status(::eufs_msgs::msg::TopicStatus::_status_type arg)
  {
    msg_.status = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::TopicStatus msg_;
};

class Init_TopicStatus_log_level
{
public:
  explicit Init_TopicStatus_log_level(::eufs_msgs::msg::TopicStatus & msg)
  : msg_(msg)
  {}
  Init_TopicStatus_status log_level(::eufs_msgs::msg::TopicStatus::_log_level_type arg)
  {
    msg_.log_level = std::move(arg);
    return Init_TopicStatus_status(msg_);
  }

private:
  ::eufs_msgs::msg::TopicStatus msg_;
};

class Init_TopicStatus_trigger_ebs
{
public:
  explicit Init_TopicStatus_trigger_ebs(::eufs_msgs::msg::TopicStatus & msg)
  : msg_(msg)
  {}
  Init_TopicStatus_log_level trigger_ebs(::eufs_msgs::msg::TopicStatus::_trigger_ebs_type arg)
  {
    msg_.trigger_ebs = std::move(arg);
    return Init_TopicStatus_log_level(msg_);
  }

private:
  ::eufs_msgs::msg::TopicStatus msg_;
};

class Init_TopicStatus_group
{
public:
  explicit Init_TopicStatus_group(::eufs_msgs::msg::TopicStatus & msg)
  : msg_(msg)
  {}
  Init_TopicStatus_trigger_ebs group(::eufs_msgs::msg::TopicStatus::_group_type arg)
  {
    msg_.group = std::move(arg);
    return Init_TopicStatus_trigger_ebs(msg_);
  }

private:
  ::eufs_msgs::msg::TopicStatus msg_;
};

class Init_TopicStatus_description
{
public:
  explicit Init_TopicStatus_description(::eufs_msgs::msg::TopicStatus & msg)
  : msg_(msg)
  {}
  Init_TopicStatus_group description(::eufs_msgs::msg::TopicStatus::_description_type arg)
  {
    msg_.description = std::move(arg);
    return Init_TopicStatus_group(msg_);
  }

private:
  ::eufs_msgs::msg::TopicStatus msg_;
};

class Init_TopicStatus_topic
{
public:
  Init_TopicStatus_topic()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_TopicStatus_description topic(::eufs_msgs::msg::TopicStatus::_topic_type arg)
  {
    msg_.topic = std::move(arg);
    return Init_TopicStatus_description(msg_);
  }

private:
  ::eufs_msgs::msg::TopicStatus msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::TopicStatus>()
{
  return eufs_msgs::msg::builder::Init_TopicStatus_topic();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__TOPIC_STATUS__BUILDER_HPP_
