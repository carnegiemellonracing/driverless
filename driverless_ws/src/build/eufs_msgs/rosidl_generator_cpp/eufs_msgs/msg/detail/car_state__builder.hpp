// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/CarState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CAR_STATE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__CAR_STATE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/car_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_CarState_state_of_charge
{
public:
  explicit Init_CarState_state_of_charge(::eufs_msgs::msg::CarState & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::CarState state_of_charge(::eufs_msgs::msg::CarState::_state_of_charge_type arg)
  {
    msg_.state_of_charge = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::CarState msg_;
};

class Init_CarState_slip_angle
{
public:
  explicit Init_CarState_slip_angle(::eufs_msgs::msg::CarState & msg)
  : msg_(msg)
  {}
  Init_CarState_state_of_charge slip_angle(::eufs_msgs::msg::CarState::_slip_angle_type arg)
  {
    msg_.slip_angle = std::move(arg);
    return Init_CarState_state_of_charge(msg_);
  }

private:
  ::eufs_msgs::msg::CarState msg_;
};

class Init_CarState_linear_acceleration_covariance
{
public:
  explicit Init_CarState_linear_acceleration_covariance(::eufs_msgs::msg::CarState & msg)
  : msg_(msg)
  {}
  Init_CarState_slip_angle linear_acceleration_covariance(::eufs_msgs::msg::CarState::_linear_acceleration_covariance_type arg)
  {
    msg_.linear_acceleration_covariance = std::move(arg);
    return Init_CarState_slip_angle(msg_);
  }

private:
  ::eufs_msgs::msg::CarState msg_;
};

class Init_CarState_linear_acceleration
{
public:
  explicit Init_CarState_linear_acceleration(::eufs_msgs::msg::CarState & msg)
  : msg_(msg)
  {}
  Init_CarState_linear_acceleration_covariance linear_acceleration(::eufs_msgs::msg::CarState::_linear_acceleration_type arg)
  {
    msg_.linear_acceleration = std::move(arg);
    return Init_CarState_linear_acceleration_covariance(msg_);
  }

private:
  ::eufs_msgs::msg::CarState msg_;
};

class Init_CarState_twist
{
public:
  explicit Init_CarState_twist(::eufs_msgs::msg::CarState & msg)
  : msg_(msg)
  {}
  Init_CarState_linear_acceleration twist(::eufs_msgs::msg::CarState::_twist_type arg)
  {
    msg_.twist = std::move(arg);
    return Init_CarState_linear_acceleration(msg_);
  }

private:
  ::eufs_msgs::msg::CarState msg_;
};

class Init_CarState_pose
{
public:
  explicit Init_CarState_pose(::eufs_msgs::msg::CarState & msg)
  : msg_(msg)
  {}
  Init_CarState_twist pose(::eufs_msgs::msg::CarState::_pose_type arg)
  {
    msg_.pose = std::move(arg);
    return Init_CarState_twist(msg_);
  }

private:
  ::eufs_msgs::msg::CarState msg_;
};

class Init_CarState_child_frame_id
{
public:
  explicit Init_CarState_child_frame_id(::eufs_msgs::msg::CarState & msg)
  : msg_(msg)
  {}
  Init_CarState_pose child_frame_id(::eufs_msgs::msg::CarState::_child_frame_id_type arg)
  {
    msg_.child_frame_id = std::move(arg);
    return Init_CarState_pose(msg_);
  }

private:
  ::eufs_msgs::msg::CarState msg_;
};

class Init_CarState_header
{
public:
  Init_CarState_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CarState_child_frame_id header(::eufs_msgs::msg::CarState::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_CarState_child_frame_id(msg_);
  }

private:
  ::eufs_msgs::msg::CarState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::CarState>()
{
  return eufs_msgs::msg::builder::Init_CarState_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__CAR_STATE__BUILDER_HPP_
