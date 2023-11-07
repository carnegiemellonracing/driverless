// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cmrdv_interfaces:msg/SimDataFrame.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__TRAITS_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__TRAITS_HPP_

#include "cmrdv_interfaces/msg/detail/sim_data_frame__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'gt_cones'
#include "eufs_msgs/msg/detail/cone_array_with_covariance__traits.hpp"
// Member 'zed_left_img'
#include "sensor_msgs/msg/detail/image__traits.hpp"
// Member 'vlp16_pts'
// Member 'zed_pts'
#include "sensor_msgs/msg/detail/point_cloud2__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<cmrdv_interfaces::msg::SimDataFrame>()
{
  return "cmrdv_interfaces::msg::SimDataFrame";
}

template<>
inline const char * name<cmrdv_interfaces::msg::SimDataFrame>()
{
  return "cmrdv_interfaces/msg/SimDataFrame";
}

template<>
struct has_fixed_size<cmrdv_interfaces::msg::SimDataFrame>
  : std::integral_constant<bool, has_fixed_size<eufs_msgs::msg::ConeArrayWithCovariance>::value && has_fixed_size<sensor_msgs::msg::Image>::value && has_fixed_size<sensor_msgs::msg::PointCloud2>::value> {};

template<>
struct has_bounded_size<cmrdv_interfaces::msg::SimDataFrame>
  : std::integral_constant<bool, has_bounded_size<eufs_msgs::msg::ConeArrayWithCovariance>::value && has_bounded_size<sensor_msgs::msg::Image>::value && has_bounded_size<sensor_msgs::msg::PointCloud2>::value> {};

template<>
struct is_message<cmrdv_interfaces::msg::SimDataFrame>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__TRAITS_HPP_
