// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cmrdv_interfaces:msg/CarROT.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__BUILDER_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__BUILDER_HPP_

#include "cmrdv_interfaces/msg/detail/car_rot__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace cmrdv_interfaces
{

namespace msg
{

namespace builder
{

class Init_CarROT_curvature
{
public:
  explicit Init_CarROT_curvature(::cmrdv_interfaces::msg::CarROT & msg)
  : msg_(msg)
  {}
  ::cmrdv_interfaces::msg::CarROT curvature(::cmrdv_interfaces::msg::CarROT::_curvature_type arg)
  {
    msg_.curvature = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cmrdv_interfaces::msg::CarROT msg_;
};

class Init_CarROT_yaw
{
public:
  explicit Init_CarROT_yaw(::cmrdv_interfaces::msg::CarROT & msg)
  : msg_(msg)
  {}
  Init_CarROT_curvature yaw(::cmrdv_interfaces::msg::CarROT::_yaw_type arg)
  {
    msg_.yaw = std::move(arg);
    return Init_CarROT_curvature(msg_);
  }

private:
  ::cmrdv_interfaces::msg::CarROT msg_;
};

class Init_CarROT_y
{
public:
  explicit Init_CarROT_y(::cmrdv_interfaces::msg::CarROT & msg)
  : msg_(msg)
  {}
  Init_CarROT_yaw y(::cmrdv_interfaces::msg::CarROT::_y_type arg)
  {
    msg_.y = std::move(arg);
    return Init_CarROT_yaw(msg_);
  }

private:
  ::cmrdv_interfaces::msg::CarROT msg_;
};

class Init_CarROT_x
{
public:
  explicit Init_CarROT_x(::cmrdv_interfaces::msg::CarROT & msg)
  : msg_(msg)
  {}
  Init_CarROT_y x(::cmrdv_interfaces::msg::CarROT::_x_type arg)
  {
    msg_.x = std::move(arg);
    return Init_CarROT_y(msg_);
  }

private:
  ::cmrdv_interfaces::msg::CarROT msg_;
};

class Init_CarROT_header
{
public:
  Init_CarROT_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CarROT_x header(::cmrdv_interfaces::msg::CarROT::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_CarROT_x(msg_);
  }

private:
  ::cmrdv_interfaces::msg::CarROT msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cmrdv_interfaces::msg::CarROT>()
{
  return cmrdv_interfaces::msg::builder::Init_CarROT_header();
}

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__BUILDER_HPP_
