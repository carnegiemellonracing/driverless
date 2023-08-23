// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from interfaces:msg/CarROT.idl
// generated code does not contain a copyright notice

#ifndef INTERFACES__MSG__DETAIL__CAR_ROT__BUILDER_HPP_
#define INTERFACES__MSG__DETAIL__CAR_ROT__BUILDER_HPP_

#include "interfaces/msg/detail/car_rot__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace interfaces
{

namespace msg
{

namespace builder
{

class Init_CarROT_curvature
{
public:
  explicit Init_CarROT_curvature(::interfaces::msg::CarROT & msg)
  : msg_(msg)
  {}
  ::interfaces::msg::CarROT curvature(::interfaces::msg::CarROT::_curvature_type arg)
  {
    msg_.curvature = std::move(arg);
    return std::move(msg_);
  }

private:
  ::interfaces::msg::CarROT msg_;
};

class Init_CarROT_yaw
{
public:
  explicit Init_CarROT_yaw(::interfaces::msg::CarROT & msg)
  : msg_(msg)
  {}
  Init_CarROT_curvature yaw(::interfaces::msg::CarROT::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_CarROT_curvature(msg_);
  }

private:
  ::interfaces::msg::CarROT msg_;
};

class Init_CarROT_y
{
public:
  explicit Init_CarROT_y(::interfaces::msg::CarROT & msg)
  : msg_(msg)
  {}
  Init_CarROT_yaw y(::interfaces::msg::CarROT::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_CarROT_yaw(msg_);
  }

private:
  ::interfaces::msg::CarROT msg_;
};

class Init_CarROT_x
{
public:
  explicit Init_CarROT_x(::interfaces::msg::CarROT & msg)
  : msg_(msg)
  {}
  Init_CarROT_y x(::interfaces::msg::CarROT::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_CarROT_y(msg_);
  }

private:
  ::interfaces::msg::CarROT msg_;
};

class Init_CarROT_header
{
public:
  Init_CarROT_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CarROT_x header(::interfaces::msg::CarROT::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_CarROT_x(msg_);
  }

private:
  ::interfaces::msg::CarROT msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::interfaces::msg::CarROT>()
{
  return interfaces::msg::builder::Init_CarROT_header();
}

}  // namespace interfaces

#endif  // INTERFACES__MSG__DETAIL__CAR_ROT__BUILDER_HPP_
