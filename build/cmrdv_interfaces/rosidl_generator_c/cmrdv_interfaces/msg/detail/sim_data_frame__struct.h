// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from cmrdv_interfaces:msg/SimDataFrame.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__STRUCT_H_
#define CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'gt_cones'
#include "eufs_msgs/msg/detail/cone_array_with_covariance__struct.h"
// Member 'zed_left_img'
#include "sensor_msgs/msg/detail/image__struct.h"
// Member 'vlp16_pts'
// Member 'zed_pts'
#include "sensor_msgs/msg/detail/point_cloud2__struct.h"

// Struct defined in msg/SimDataFrame in the package cmrdv_interfaces.
typedef struct cmrdv_interfaces__msg__SimDataFrame
{
  eufs_msgs__msg__ConeArrayWithCovariance gt_cones;
  sensor_msgs__msg__Image zed_left_img;
  sensor_msgs__msg__PointCloud2 vlp16_pts;
  sensor_msgs__msg__PointCloud2 zed_pts;
} cmrdv_interfaces__msg__SimDataFrame;

// Struct for a sequence of cmrdv_interfaces__msg__SimDataFrame.
typedef struct cmrdv_interfaces__msg__SimDataFrame__Sequence
{
  cmrdv_interfaces__msg__SimDataFrame * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} cmrdv_interfaces__msg__SimDataFrame__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__SIM_DATA_FRAME__STRUCT_H_
