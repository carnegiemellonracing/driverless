// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/LapStats.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__LAP_STATS__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__LAP_STATS__BUILDER_HPP_

#include "eufs_msgs/msg/detail/lap_stats__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_LapStats_max_deviation
{
public:
  explicit Init_LapStats_max_deviation(::eufs_msgs::msg::LapStats & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::LapStats max_deviation(::eufs_msgs::msg::LapStats::_max_deviation_type arg)
  {
    msg_.max_deviation = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::LapStats msg_;
};

class Init_LapStats_deviation_var
{
public:
  explicit Init_LapStats_deviation_var(::eufs_msgs::msg::LapStats & msg)
  : msg_(msg)
  {}
  Init_LapStats_max_deviation deviation_var(::eufs_msgs::msg::LapStats::_deviation_var_type arg)
  {
    msg_.deviation_var = std::move(arg);
    return Init_LapStats_max_deviation(msg_);
  }

private:
  ::eufs_msgs::msg::LapStats msg_;
};

class Init_LapStats_normalized_deviation_mse
{
public:
  explicit Init_LapStats_normalized_deviation_mse(::eufs_msgs::msg::LapStats & msg)
  : msg_(msg)
  {}
  Init_LapStats_deviation_var normalized_deviation_mse(::eufs_msgs::msg::LapStats::_normalized_deviation_mse_type arg)
  {
    msg_.normalized_deviation_mse = std::move(arg);
    return Init_LapStats_deviation_var(msg_);
  }

private:
  ::eufs_msgs::msg::LapStats msg_;
};

class Init_LapStats_max_lateral_accel
{
public:
  explicit Init_LapStats_max_lateral_accel(::eufs_msgs::msg::LapStats & msg)
  : msg_(msg)
  {}
  Init_LapStats_normalized_deviation_mse max_lateral_accel(::eufs_msgs::msg::LapStats::_max_lateral_accel_type arg)
  {
    msg_.max_lateral_accel = std::move(arg);
    return Init_LapStats_normalized_deviation_mse(msg_);
  }

private:
  ::eufs_msgs::msg::LapStats msg_;
};

class Init_LapStats_max_slip
{
public:
  explicit Init_LapStats_max_slip(::eufs_msgs::msg::LapStats & msg)
  : msg_(msg)
  {}
  Init_LapStats_max_lateral_accel max_slip(::eufs_msgs::msg::LapStats::_max_slip_type arg)
  {
    msg_.max_slip = std::move(arg);
    return Init_LapStats_max_lateral_accel(msg_);
  }

private:
  ::eufs_msgs::msg::LapStats msg_;
};

class Init_LapStats_speed_var
{
public:
  explicit Init_LapStats_speed_var(::eufs_msgs::msg::LapStats & msg)
  : msg_(msg)
  {}
  Init_LapStats_max_slip speed_var(::eufs_msgs::msg::LapStats::_speed_var_type arg)
  {
    msg_.speed_var = std::move(arg);
    return Init_LapStats_max_slip(msg_);
  }

private:
  ::eufs_msgs::msg::LapStats msg_;
};

class Init_LapStats_max_speed
{
public:
  explicit Init_LapStats_max_speed(::eufs_msgs::msg::LapStats & msg)
  : msg_(msg)
  {}
  Init_LapStats_speed_var max_speed(::eufs_msgs::msg::LapStats::_max_speed_type arg)
  {
    msg_.max_speed = std::move(arg);
    return Init_LapStats_speed_var(msg_);
  }

private:
  ::eufs_msgs::msg::LapStats msg_;
};

class Init_LapStats_avg_speed
{
public:
  explicit Init_LapStats_avg_speed(::eufs_msgs::msg::LapStats & msg)
  : msg_(msg)
  {}
  Init_LapStats_max_speed avg_speed(::eufs_msgs::msg::LapStats::_avg_speed_type arg)
  {
    msg_.avg_speed = std::move(arg);
    return Init_LapStats_max_speed(msg_);
  }

private:
  ::eufs_msgs::msg::LapStats msg_;
};

class Init_LapStats_lap_time
{
public:
  explicit Init_LapStats_lap_time(::eufs_msgs::msg::LapStats & msg)
  : msg_(msg)
  {}
  Init_LapStats_avg_speed lap_time(::eufs_msgs::msg::LapStats::_lap_time_type arg)
  {
    msg_.lap_time = std::move(arg);
    return Init_LapStats_avg_speed(msg_);
  }

private:
  ::eufs_msgs::msg::LapStats msg_;
};

class Init_LapStats_lap_number
{
public:
  explicit Init_LapStats_lap_number(::eufs_msgs::msg::LapStats & msg)
  : msg_(msg)
  {}
  Init_LapStats_lap_time lap_number(::eufs_msgs::msg::LapStats::_lap_number_type arg)
  {
    msg_.lap_number = std::move(arg);
    return Init_LapStats_lap_time(msg_);
  }

private:
  ::eufs_msgs::msg::LapStats msg_;
};

class Init_LapStats_header
{
public:
  Init_LapStats_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_LapStats_lap_number header(::eufs_msgs::msg::LapStats::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_LapStats_lap_number(msg_);
  }

private:
  ::eufs_msgs::msg::LapStats msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::LapStats>()
{
  return eufs_msgs::msg::builder::Init_LapStats_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__LAP_STATS__BUILDER_HPP_
