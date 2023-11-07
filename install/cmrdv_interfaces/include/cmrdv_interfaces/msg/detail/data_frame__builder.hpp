// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cmrdv_interfaces:msg/DataFrame.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__BUILDER_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__BUILDER_HPP_

#include "cmrdv_interfaces/msg/detail/data_frame__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace cmrdv_interfaces
{

namespace msg
{

namespace builder
{

class Init_DataFrame_imu
{
public:
  explicit Init_DataFrame_imu(::cmrdv_interfaces::msg::DataFrame & msg)
  : msg_(msg)
  {}
  ::cmrdv_interfaces::msg::DataFrame imu(::cmrdv_interfaces::msg::DataFrame::_imu_type arg)
  {
    msg_.imu = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cmrdv_interfaces::msg::DataFrame msg_;
};

class Init_DataFrame_sbg
{
public:
  explicit Init_DataFrame_sbg(::cmrdv_interfaces::msg::DataFrame & msg)
  : msg_(msg)
  {}
  Init_DataFrame_imu sbg(::cmrdv_interfaces::msg::DataFrame::_sbg_type arg)
  {
    msg_.sbg = std::move(arg);
    return Init_DataFrame_imu(msg_);
  }

private:
  ::cmrdv_interfaces::msg::DataFrame msg_;
};

class Init_DataFrame_zed_pts
{
public:
  explicit Init_DataFrame_zed_pts(::cmrdv_interfaces::msg::DataFrame & msg)
  : msg_(msg)
  {}
  Init_DataFrame_sbg zed_pts(::cmrdv_interfaces::msg::DataFrame::_zed_pts_type arg)
  {
    msg_.zed_pts = std::move(arg);
    return Init_DataFrame_sbg(msg_);
  }

private:
  ::cmrdv_interfaces::msg::DataFrame msg_;
};

class Init_DataFrame_zed_left_img
{
public:
  Init_DataFrame_zed_left_img()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_DataFrame_zed_pts zed_left_img(::cmrdv_interfaces::msg::DataFrame::_zed_left_img_type arg)
  {
    msg_.zed_left_img = std::move(arg);
    return Init_DataFrame_zed_pts(msg_);
  }

private:
  ::cmrdv_interfaces::msg::DataFrame msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cmrdv_interfaces::msg::DataFrame>()
{
  return cmrdv_interfaces::msg::builder::Init_DataFrame_zed_left_img();
}

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__BUILDER_HPP_
