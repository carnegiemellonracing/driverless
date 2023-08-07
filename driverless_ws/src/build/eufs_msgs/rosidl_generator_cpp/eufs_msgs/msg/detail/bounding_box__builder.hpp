// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/BoundingBox.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__BOUNDING_BOX__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__BOUNDING_BOX__BUILDER_HPP_

#include "eufs_msgs/msg/detail/bounding_box__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_BoundingBox_ymax
{
public:
  explicit Init_BoundingBox_ymax(::eufs_msgs::msg::BoundingBox & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::BoundingBox ymax(::eufs_msgs::msg::BoundingBox::_ymax_type arg)
  {
    msg_.ymax = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::BoundingBox msg_;
};

class Init_BoundingBox_xmax
{
public:
  explicit Init_BoundingBox_xmax(::eufs_msgs::msg::BoundingBox & msg)
  : msg_(msg)
  {}
  Init_BoundingBox_ymax xmax(::eufs_msgs::msg::BoundingBox::_xmax_type arg)
  {
    msg_.xmax = std::move(arg);
    return Init_BoundingBox_ymax(msg_);
  }

private:
  ::eufs_msgs::msg::BoundingBox msg_;
};

class Init_BoundingBox_ymin
{
public:
  explicit Init_BoundingBox_ymin(::eufs_msgs::msg::BoundingBox & msg)
  : msg_(msg)
  {}
  Init_BoundingBox_xmax ymin(::eufs_msgs::msg::BoundingBox::_ymin_type arg)
  {
    msg_.ymin = std::move(arg);
    return Init_BoundingBox_xmax(msg_);
  }

private:
  ::eufs_msgs::msg::BoundingBox msg_;
};

class Init_BoundingBox_xmin
{
public:
  explicit Init_BoundingBox_xmin(::eufs_msgs::msg::BoundingBox & msg)
  : msg_(msg)
  {}
  Init_BoundingBox_ymin xmin(::eufs_msgs::msg::BoundingBox::_xmin_type arg)
  {
    msg_.xmin = std::move(arg);
    return Init_BoundingBox_ymin(msg_);
  }

private:
  ::eufs_msgs::msg::BoundingBox msg_;
};

class Init_BoundingBox_type
{
public:
  explicit Init_BoundingBox_type(::eufs_msgs::msg::BoundingBox & msg)
  : msg_(msg)
  {}
  Init_BoundingBox_xmin type(::eufs_msgs::msg::BoundingBox::_type_type arg)
  {
    msg_.type = std::move(arg);
    return Init_BoundingBox_xmin(msg_);
  }

private:
  ::eufs_msgs::msg::BoundingBox msg_;
};

class Init_BoundingBox_probability
{
public:
  explicit Init_BoundingBox_probability(::eufs_msgs::msg::BoundingBox & msg)
  : msg_(msg)
  {}
  Init_BoundingBox_type probability(::eufs_msgs::msg::BoundingBox::_probability_type arg)
  {
    msg_.probability = std::move(arg);
    return Init_BoundingBox_type(msg_);
  }

private:
  ::eufs_msgs::msg::BoundingBox msg_;
};

class Init_BoundingBox_color
{
public:
  Init_BoundingBox_color()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_BoundingBox_probability color(::eufs_msgs::msg::BoundingBox::_color_type arg)
  {
    msg_.color = std::move(arg);
    return Init_BoundingBox_probability(msg_);
  }

private:
  ::eufs_msgs::msg::BoundingBox msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::BoundingBox>()
{
  return eufs_msgs::msg::builder::Init_BoundingBox_color();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__BOUNDING_BOX__BUILDER_HPP_
