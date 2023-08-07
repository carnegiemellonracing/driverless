// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/EKFState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__EKF_STATE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__EKF_STATE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/ekf_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_EKFState_recommends_failure
{
public:
  explicit Init_EKFState_recommends_failure(::eufs_msgs::msg::EKFState & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::EKFState recommends_failure(::eufs_msgs::msg::EKFState::_recommends_failure_type arg)
  {
    msg_.recommends_failure = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::EKFState msg_;
};

class Init_EKFState_consecutive_turns_over_covariance_limit
{
public:
  explicit Init_EKFState_consecutive_turns_over_covariance_limit(::eufs_msgs::msg::EKFState & msg)
  : msg_(msg)
  {}
  Init_EKFState_recommends_failure consecutive_turns_over_covariance_limit(::eufs_msgs::msg::EKFState::_consecutive_turns_over_covariance_limit_type arg)
  {
    msg_.consecutive_turns_over_covariance_limit = std::move(arg);
    return Init_EKFState_recommends_failure(msg_);
  }

private:
  ::eufs_msgs::msg::EKFState msg_;
};

class Init_EKFState_currently_over_covariance_limit
{
public:
  explicit Init_EKFState_currently_over_covariance_limit(::eufs_msgs::msg::EKFState & msg)
  : msg_(msg)
  {}
  Init_EKFState_consecutive_turns_over_covariance_limit currently_over_covariance_limit(::eufs_msgs::msg::EKFState::_currently_over_covariance_limit_type arg)
  {
    msg_.currently_over_covariance_limit = std::move(arg);
    return Init_EKFState_consecutive_turns_over_covariance_limit(msg_);
  }

private:
  ::eufs_msgs::msg::EKFState msg_;
};

class Init_EKFState_ekf_accel_received
{
public:
  explicit Init_EKFState_ekf_accel_received(::eufs_msgs::msg::EKFState & msg)
  : msg_(msg)
  {}
  Init_EKFState_currently_over_covariance_limit ekf_accel_received(::eufs_msgs::msg::EKFState::_ekf_accel_received_type arg)
  {
    msg_.ekf_accel_received = std::move(arg);
    return Init_EKFState_currently_over_covariance_limit(msg_);
  }

private:
  ::eufs_msgs::msg::EKFState msg_;
};

class Init_EKFState_ekf_odom_received
{
public:
  explicit Init_EKFState_ekf_odom_received(::eufs_msgs::msg::EKFState & msg)
  : msg_(msg)
  {}
  Init_EKFState_ekf_accel_received ekf_odom_received(::eufs_msgs::msg::EKFState::_ekf_odom_received_type arg)
  {
    msg_.ekf_odom_received = std::move(arg);
    return Init_EKFState_ekf_accel_received(msg_);
  }

private:
  ::eufs_msgs::msg::EKFState msg_;
};

class Init_EKFState_wheel_odom_received
{
public:
  explicit Init_EKFState_wheel_odom_received(::eufs_msgs::msg::EKFState & msg)
  : msg_(msg)
  {}
  Init_EKFState_ekf_odom_received wheel_odom_received(::eufs_msgs::msg::EKFState::_wheel_odom_received_type arg)
  {
    msg_.wheel_odom_received = std::move(arg);
    return Init_EKFState_ekf_odom_received(msg_);
  }

private:
  ::eufs_msgs::msg::EKFState msg_;
};

class Init_EKFState_imu_received
{
public:
  explicit Init_EKFState_imu_received(::eufs_msgs::msg::EKFState & msg)
  : msg_(msg)
  {}
  Init_EKFState_wheel_odom_received imu_received(::eufs_msgs::msg::EKFState::_imu_received_type arg)
  {
    msg_.imu_received = std::move(arg);
    return Init_EKFState_wheel_odom_received(msg_);
  }

private:
  ::eufs_msgs::msg::EKFState msg_;
};

class Init_EKFState_gps_received
{
public:
  Init_EKFState_gps_received()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_EKFState_imu_received gps_received(::eufs_msgs::msg::EKFState::_gps_received_type arg)
  {
    msg_.gps_received = std::move(arg);
    return Init_EKFState_imu_received(msg_);
  }

private:
  ::eufs_msgs::msg::EKFState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::EKFState>()
{
  return eufs_msgs::msg::builder::Init_EKFState_gps_received();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__EKF_STATE__BUILDER_HPP_
