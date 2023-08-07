// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/Costmap.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__COSTMAP__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__COSTMAP__BUILDER_HPP_

#include "eufs_msgs/msg/detail/costmap__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_Costmap_channel3
{
public:
  explicit Init_Costmap_channel3(::eufs_msgs::msg::Costmap & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::Costmap channel3(::eufs_msgs::msg::Costmap::_channel3_type arg)
  {
    msg_.channel3 = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::Costmap msg_;
};

class Init_Costmap_channel2
{
public:
  explicit Init_Costmap_channel2(::eufs_msgs::msg::Costmap & msg)
  : msg_(msg)
  {}
  Init_Costmap_channel3 channel2(::eufs_msgs::msg::Costmap::_channel2_type arg)
  {
    msg_.channel2 = std::move(arg);
    return Init_Costmap_channel3(msg_);
  }

private:
  ::eufs_msgs::msg::Costmap msg_;
};

class Init_Costmap_channel1
{
public:
  explicit Init_Costmap_channel1(::eufs_msgs::msg::Costmap & msg)
  : msg_(msg)
  {}
  Init_Costmap_channel2 channel1(::eufs_msgs::msg::Costmap::_channel1_type arg)
  {
    msg_.channel1 = std::move(arg);
    return Init_Costmap_channel2(msg_);
  }

private:
  ::eufs_msgs::msg::Costmap msg_;
};

class Init_Costmap_channel0
{
public:
  explicit Init_Costmap_channel0(::eufs_msgs::msg::Costmap & msg)
  : msg_(msg)
  {}
  Init_Costmap_channel1 channel0(::eufs_msgs::msg::Costmap::_channel0_type arg)
  {
    msg_.channel0 = std::move(arg);
    return Init_Costmap_channel1(msg_);
  }

private:
  ::eufs_msgs::msg::Costmap msg_;
};

class Init_Costmap_pixels_per_meter
{
public:
  explicit Init_Costmap_pixels_per_meter(::eufs_msgs::msg::Costmap & msg)
  : msg_(msg)
  {}
  Init_Costmap_channel0 pixels_per_meter(::eufs_msgs::msg::Costmap::_pixels_per_meter_type arg)
  {
    msg_.pixels_per_meter = std::move(arg);
    return Init_Costmap_channel0(msg_);
  }

private:
  ::eufs_msgs::msg::Costmap msg_;
};

class Init_Costmap_y_bounds_max
{
public:
  explicit Init_Costmap_y_bounds_max(::eufs_msgs::msg::Costmap & msg)
  : msg_(msg)
  {}
  Init_Costmap_pixels_per_meter y_bounds_max(::eufs_msgs::msg::Costmap::_y_bounds_max_type arg)
  {
    msg_.y_bounds_max = std::move(arg);
    return Init_Costmap_pixels_per_meter(msg_);
  }

private:
  ::eufs_msgs::msg::Costmap msg_;
};

class Init_Costmap_y_bounds_min
{
public:
  explicit Init_Costmap_y_bounds_min(::eufs_msgs::msg::Costmap & msg)
  : msg_(msg)
  {}
  Init_Costmap_y_bounds_max y_bounds_min(::eufs_msgs::msg::Costmap::_y_bounds_min_type arg)
  {
    msg_.y_bounds_min = std::move(arg);
    return Init_Costmap_y_bounds_max(msg_);
  }

private:
  ::eufs_msgs::msg::Costmap msg_;
};

class Init_Costmap_x_bounds_max
{
public:
  explicit Init_Costmap_x_bounds_max(::eufs_msgs::msg::Costmap & msg)
  : msg_(msg)
  {}
  Init_Costmap_y_bounds_min x_bounds_max(::eufs_msgs::msg::Costmap::_x_bounds_max_type arg)
  {
    msg_.x_bounds_max = std::move(arg);
    return Init_Costmap_y_bounds_min(msg_);
  }

private:
  ::eufs_msgs::msg::Costmap msg_;
};

class Init_Costmap_x_bounds_min
{
public:
  Init_Costmap_x_bounds_min()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_Costmap_x_bounds_max x_bounds_min(::eufs_msgs::msg::Costmap::_x_bounds_min_type arg)
  {
    msg_.x_bounds_min = std::move(arg);
    return Init_Costmap_x_bounds_max(msg_);
  }

private:
  ::eufs_msgs::msg::Costmap msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::Costmap>()
{
  return eufs_msgs::msg::builder::Init_Costmap_x_bounds_min();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__COSTMAP__BUILDER_HPP_
