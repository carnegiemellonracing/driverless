// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from cmrdv_interfaces:msg/ConeList.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CONE_LIST__BUILDER_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__CONE_LIST__BUILDER_HPP_

#include "cmrdv_interfaces/msg/detail/cone_list__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace cmrdv_interfaces
{

namespace msg
{

namespace builder
{

class Init_ConeList_orange_cones
{
public:
  explicit Init_ConeList_orange_cones(::cmrdv_interfaces::msg::ConeList & msg)
  : msg_(msg)
  {}
  ::cmrdv_interfaces::msg::ConeList orange_cones(::cmrdv_interfaces::msg::ConeList::_orange_cones_type arg)
  {
    msg_.orange_cones = std::move(arg);
    return std::move(msg_);
  }

private:
  ::cmrdv_interfaces::msg::ConeList msg_;
};

class Init_ConeList_yellow_cones
{
public:
  explicit Init_ConeList_yellow_cones(::cmrdv_interfaces::msg::ConeList & msg)
  : msg_(msg)
  {}
  Init_ConeList_orange_cones yellow_cones(::cmrdv_interfaces::msg::ConeList::_yellow_cones_type arg)
  {
    msg_.yellow_cones = std::move(arg);
    return Init_ConeList_orange_cones(msg_);
  }

private:
  ::cmrdv_interfaces::msg::ConeList msg_;
};

class Init_ConeList_blue_cones
{
public:
  Init_ConeList_blue_cones()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_ConeList_yellow_cones blue_cones(::cmrdv_interfaces::msg::ConeList::_blue_cones_type arg)
  {
    msg_.blue_cones = std::move(arg);
    return Init_ConeList_yellow_cones(msg_);
  }

private:
  ::cmrdv_interfaces::msg::ConeList msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::cmrdv_interfaces::msg::ConeList>()
{
  return cmrdv_interfaces::msg::builder::Init_ConeList_blue_cones();
}

}  // namespace cmrdv_interfaces

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CONE_LIST__BUILDER_HPP_
