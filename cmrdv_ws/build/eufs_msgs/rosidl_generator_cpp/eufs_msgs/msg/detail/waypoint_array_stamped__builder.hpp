// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/WaypointArrayStamped.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WAYPOINT_ARRAY_STAMPED__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__WAYPOINT_ARRAY_STAMPED__BUILDER_HPP_

#include "eufs_msgs/msg/detail/waypoint_array_stamped__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_WaypointArrayStamped_waypoints
{
public:
  explicit Init_WaypointArrayStamped_waypoints(::eufs_msgs::msg::WaypointArrayStamped & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::WaypointArrayStamped waypoints(::eufs_msgs::msg::WaypointArrayStamped::_waypoints_type arg)
  {
    msg_.waypoints = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::WaypointArrayStamped msg_;
};

class Init_WaypointArrayStamped_header
{
public:
  Init_WaypointArrayStamped_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_WaypointArrayStamped_waypoints header(::eufs_msgs::msg::WaypointArrayStamped::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_WaypointArrayStamped_waypoints(msg_);
  }

private:
  ::eufs_msgs::msg::WaypointArrayStamped msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::WaypointArrayStamped>()
{
  return eufs_msgs::msg::builder::Init_WaypointArrayStamped_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__WAYPOINT_ARRAY_STAMPED__BUILDER_HPP_
