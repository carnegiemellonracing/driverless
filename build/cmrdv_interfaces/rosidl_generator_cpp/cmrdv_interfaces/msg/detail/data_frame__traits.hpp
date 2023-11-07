// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from cmrdv_interfaces:msg/DataFrame.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__TRAITS_HPP_
#define CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__TRAITS_HPP_

#include "cmrdv_interfaces/msg/detail/data_frame__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'zed_left_img'
#include "sensor_msgs/msg/detail/image__traits.hpp"
// Member 'zed_pts'
#include "sensor_msgs/msg/detail/point_cloud2__traits.hpp"
// Member 'sbg'
#include "sbg_driver/msg/detail/sbg_gps_pos__traits.hpp"
// Member 'imu'
#include "sbg_driver/msg/detail/sbg_imu_data__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<cmrdv_interfaces::msg::DataFrame>()
{
  return "cmrdv_interfaces::msg::DataFrame";
}

template<>
inline const char * name<cmrdv_interfaces::msg::DataFrame>()
{
  return "cmrdv_interfaces/msg/DataFrame";
}

template<>
struct has_fixed_size<cmrdv_interfaces::msg::DataFrame>
  : std::integral_constant<bool, has_fixed_size<sbg_driver::msg::SbgGpsPos>::value && has_fixed_size<sbg_driver::msg::SbgImuData>::value && has_fixed_size<sensor_msgs::msg::Image>::value && has_fixed_size<sensor_msgs::msg::PointCloud2>::value> {};

template<>
struct has_bounded_size<cmrdv_interfaces::msg::DataFrame>
  : std::integral_constant<bool, has_bounded_size<sbg_driver::msg::SbgGpsPos>::value && has_bounded_size<sbg_driver::msg::SbgImuData>::value && has_bounded_size<sensor_msgs::msg::Image>::value && has_bounded_size<sensor_msgs::msg::PointCloud2>::value> {};

template<>
struct is_message<cmrdv_interfaces::msg::DataFrame>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__TRAITS_HPP_
