// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/PathIntegralParams.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__BUILDER_HPP_

#include "eufs_msgs/msg/detail/path_integral_params__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_PathIntegralParams_desired_speed
{
public:
  explicit Init_PathIntegralParams_desired_speed(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::PathIntegralParams desired_speed(::eufs_msgs::msg::PathIntegralParams::_desired_speed_type arg)
  {
    msg_.desired_speed = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_map_path
{
public:
  explicit Init_PathIntegralParams_map_path(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_desired_speed map_path(::eufs_msgs::msg::PathIntegralParams::_map_path_type arg)
  {
    msg_.map_path = std::move(arg);
    return Init_PathIntegralParams_desired_speed(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_crash_coeff
{
public:
  explicit Init_PathIntegralParams_crash_coeff(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_map_path crash_coeff(::eufs_msgs::msg::PathIntegralParams::_crash_coeff_type arg)
  {
    msg_.crash_coeff = std::move(arg);
    return Init_PathIntegralParams_map_path(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_track_slop
{
public:
  explicit Init_PathIntegralParams_track_slop(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_crash_coeff track_slop(::eufs_msgs::msg::PathIntegralParams::_track_slop_type arg)
  {
    msg_.track_slop = std::move(arg);
    return Init_PathIntegralParams_crash_coeff(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_max_slip_angle
{
public:
  explicit Init_PathIntegralParams_max_slip_angle(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_track_slop max_slip_angle(::eufs_msgs::msg::PathIntegralParams::_max_slip_angle_type arg)
  {
    msg_.max_slip_angle = std::move(arg);
    return Init_PathIntegralParams_track_slop(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_track_coefficient
{
public:
  explicit Init_PathIntegralParams_track_coefficient(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_max_slip_angle track_coefficient(::eufs_msgs::msg::PathIntegralParams::_track_coefficient_type arg)
  {
    msg_.track_coefficient = std::move(arg);
    return Init_PathIntegralParams_max_slip_angle(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_speed_coefficient
{
public:
  explicit Init_PathIntegralParams_speed_coefficient(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_track_coefficient speed_coefficient(::eufs_msgs::msg::PathIntegralParams::_speed_coefficient_type arg)
  {
    msg_.speed_coefficient = std::move(arg);
    return Init_PathIntegralParams_track_coefficient(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_max_throttle
{
public:
  explicit Init_PathIntegralParams_max_throttle(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_speed_coefficient max_throttle(::eufs_msgs::msg::PathIntegralParams::_max_throttle_type arg)
  {
    msg_.max_throttle = std::move(arg);
    return Init_PathIntegralParams_speed_coefficient(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_throttle_var
{
public:
  explicit Init_PathIntegralParams_throttle_var(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_max_throttle throttle_var(::eufs_msgs::msg::PathIntegralParams::_throttle_var_type arg)
  {
    msg_.throttle_var = std::move(arg);
    return Init_PathIntegralParams_max_throttle(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_steering_var
{
public:
  explicit Init_PathIntegralParams_steering_var(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_throttle_var steering_var(::eufs_msgs::msg::PathIntegralParams::_steering_var_type arg)
  {
    msg_.steering_var = std::move(arg);
    return Init_PathIntegralParams_throttle_var(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_init_throttle
{
public:
  explicit Init_PathIntegralParams_init_throttle(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_steering_var init_throttle(::eufs_msgs::msg::PathIntegralParams::_init_throttle_type arg)
  {
    msg_.init_throttle = std::move(arg);
    return Init_PathIntegralParams_steering_var(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_init_steering
{
public:
  explicit Init_PathIntegralParams_init_steering(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_init_throttle init_steering(::eufs_msgs::msg::PathIntegralParams::_init_steering_type arg)
  {
    msg_.init_steering = std::move(arg);
    return Init_PathIntegralParams_init_throttle(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_gamma
{
public:
  explicit Init_PathIntegralParams_gamma(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_init_steering gamma(::eufs_msgs::msg::PathIntegralParams::_gamma_type arg)
  {
    msg_.gamma = std::move(arg);
    return Init_PathIntegralParams_init_steering(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_num_iters
{
public:
  explicit Init_PathIntegralParams_num_iters(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_gamma num_iters(::eufs_msgs::msg::PathIntegralParams::_num_iters_type arg)
  {
    msg_.num_iters = std::move(arg);
    return Init_PathIntegralParams_gamma(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_num_timesteps
{
public:
  explicit Init_PathIntegralParams_num_timesteps(::eufs_msgs::msg::PathIntegralParams & msg)
  : msg_(msg)
  {}
  Init_PathIntegralParams_num_iters num_timesteps(::eufs_msgs::msg::PathIntegralParams::_num_timesteps_type arg)
  {
    msg_.num_timesteps = std::move(arg);
    return Init_PathIntegralParams_num_iters(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

class Init_PathIntegralParams_hz
{
public:
  Init_PathIntegralParams_hz()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_PathIntegralParams_num_timesteps hz(::eufs_msgs::msg::PathIntegralParams::_hz_type arg)
  {
    msg_.hz = std::move(arg);
    return Init_PathIntegralParams_num_timesteps(msg_);
  }

private:
  ::eufs_msgs::msg::PathIntegralParams msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::PathIntegralParams>()
{
  return eufs_msgs::msg::builder::Init_PathIntegralParams_hz();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__PATH_INTEGRAL_PARAMS__BUILDER_HPP_
