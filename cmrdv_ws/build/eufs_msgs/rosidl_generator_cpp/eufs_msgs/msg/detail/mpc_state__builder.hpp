// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/MPCState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__MPC_STATE__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__MPC_STATE__BUILDER_HPP_

#include "eufs_msgs/msg/detail/mpc_state__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_MPCState_cost
{
public:
  explicit Init_MPCState_cost(::eufs_msgs::msg::MPCState & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::MPCState cost(::eufs_msgs::msg::MPCState::_cost_type arg)
  {
    msg_.cost = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::MPCState msg_;
};

class Init_MPCState_solve_time
{
public:
  explicit Init_MPCState_solve_time(::eufs_msgs::msg::MPCState & msg)
  : msg_(msg)
  {}
  Init_MPCState_cost solve_time(::eufs_msgs::msg::MPCState::_solve_time_type arg)
  {
    msg_.solve_time = std::move(arg);
    return Init_MPCState_cost(msg_);
  }

private:
  ::eufs_msgs::msg::MPCState msg_;
};

class Init_MPCState_iterations
{
public:
  explicit Init_MPCState_iterations(::eufs_msgs::msg::MPCState & msg)
  : msg_(msg)
  {}
  Init_MPCState_solve_time iterations(::eufs_msgs::msg::MPCState::_iterations_type arg)
  {
    msg_.iterations = std::move(arg);
    return Init_MPCState_solve_time(msg_);
  }

private:
  ::eufs_msgs::msg::MPCState msg_;
};

class Init_MPCState_exitflag
{
public:
  Init_MPCState_exitflag()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MPCState_iterations exitflag(::eufs_msgs::msg::MPCState::_exitflag_type arg)
  {
    msg_.exitflag = std::move(arg);
    return Init_MPCState_iterations(msg_);
  }

private:
  ::eufs_msgs::msg::MPCState msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::MPCState>()
{
  return eufs_msgs::msg::builder::Init_MPCState_exitflag();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__MPC_STATE__BUILDER_HPP_
