// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cmrdv_interfaces:msg/SimDataFrame.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__BUILDER_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__BUILDER_HPP_

#include "cmrdv_interfaces/msg/detail/sim_data_frame__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace cmrdv_interfaces
{

namespace msg
{

namespace builder
{

class Init_SimDataFrame_zed_pts
{
public:
  explicit Init_SimDataFrame_zed_pts(::cmrdv_interfaces::msg::SimDataFrame & msg)
  : msg_(msg)
  {}
  ::cmrdv_interfaces::msg::SimDataFrame zed_pts(::cmrdv_interfaces::msg::SimDataFrame::_zed_pts_type arg)
  {
    msg_.zed_pts = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cmrdv_interfaces::msg::SimDataFrame msg_;
};

class Init_SimDataFrame_vlp16_pts
{
public:
  explicit Init_SimDataFrame_vlp16_pts(::cmrdv_interfaces::msg::SimDataFrame & msg)
  : msg_(msg)
  {}
  Init_SimDataFrame_zed_pts vlp16_pts(::cmrdv_interfaces::msg::SimDataFrame::_vlp16_pts_type arg)
  {
    msg_.vlp16_pts = std::move(arg);
    return Init_SimDataFrame_zed_pts(msg_);
  }

private:
  ::cmrdv_interfaces::msg::SimDataFrame msg_;
};

class Init_SimDataFrame_zed_left_img
{
public:
  explicit Init_SimDataFrame_zed_left_img(::cmrdv_interfaces::msg::SimDataFrame & msg)
  : msg_(msg)
  {}
  Init_SimDataFrame_vlp16_pts zed_left_img(::cmrdv_interfaces::msg::SimDataFrame::_zed_left_img_type arg)
  {
    msg_.zed_left_img = std::move(arg);
    return Init_SimDataFrame_vlp16_pts(msg_);
  }

private:
  ::cmrdv_interfaces::msg::SimDataFrame msg_;
};

class Init_SimDataFrame_gt_cones
{
public:
  Init_SimDataFrame_gt_cones()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SimDataFrame_zed_left_img gt_cones(::cmrdv_interfaces::msg::SimDataFrame::_gt_cones_type arg)
  {
    msg_.gt_cones = std::move(arg);
    return Init_SimDataFrame_zed_left_img(msg_);
  }

private:
  ::cmrdv_interfaces::msg::SimDataFrame msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cmrdv_interfaces::msg::SimDataFrame>()
{
  return cmrdv_interfaces::msg::builder::Init_SimDataFrame_gt_cones();
}

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__BUILDER_HPP_
