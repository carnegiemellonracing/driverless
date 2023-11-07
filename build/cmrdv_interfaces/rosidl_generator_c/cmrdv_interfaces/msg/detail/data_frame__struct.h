// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cmrdv_interfaces:msg/DataFrame.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__STRUCT_H_
#define CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'zed_left_img'
#include "sensor_msgs/msg/detail/image__struct.h"
// Member 'zed_pts'
#include "sensor_msgs/msg/detail/point_cloud2__struct.h"
// Member 'sbg'
#include "sbg_driver/msg/detail/sbg_gps_pos__struct.h"
// Member 'imu'
#include "sbg_driver/msg/detail/sbg_imu_data__struct.h"

// Struct defined in msg/DataFrame in the package cmrdv_interfaces.
typedef struct cmrdv_interfaces__msg__DataFrame
{
  sensor_msgs__msg__Image zed_left_img;
  sensor_msgs__msg__PointCloud2 zed_pts;
  sbg_driver__msg__SbgGpsPos sbg;
  sbg_driver__msg__SbgImuData imu;
} cmrdv_interfaces__msg__DataFrame;

// Struct for a sequence of cmrdv_interfaces__msg__DataFrame.
typedef struct cmrdv_interfaces__msg__DataFrame__Sequence
{
  cmrdv_interfaces__msg__DataFrame * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cmrdv_interfaces__msg__DataFrame__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__STRUCT_H_
