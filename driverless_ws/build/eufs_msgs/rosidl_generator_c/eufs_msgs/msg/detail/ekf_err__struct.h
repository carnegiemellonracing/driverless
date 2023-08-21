// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/EKFErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__EKF_ERR__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__EKF_ERR__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

// Struct defined in msg/EKFErr in the package eufs_msgs.
typedef struct eufs_msgs__msg__EKFErr
{
  std_msgs__msg__Header header;
  double gps_x_vel_err;
  double gps_y_vel_err;
  double imu_x_acc_err;
  double imu_y_acc_err;
  double imu_yaw_err;
  double ekf_x_vel_var;
  double ekf_y_vel_var;
  double ekf_x_acc_var;
  double ekf_y_acc_var;
  double ekf_yaw_var;
} eufs_msgs__msg__EKFErr;

// Struct for a sequence of eufs_msgs__msg__EKFErr.
typedef struct eufs_msgs__msg__EKFErr__Sequence
{
  eufs_msgs__msg__EKFErr * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__EKFErr__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__EKF_ERR__STRUCT_H_
