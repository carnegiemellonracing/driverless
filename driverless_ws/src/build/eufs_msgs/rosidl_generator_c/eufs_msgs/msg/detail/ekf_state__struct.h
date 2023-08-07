// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/EKFState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__EKF_STATE__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__EKF_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Struct defined in msg/EKFState in the package eufs_msgs.
typedef struct eufs_msgs__msg__EKFState
{
  bool gps_received;
  bool imu_received;
  bool wheel_odom_received;
  bool ekf_odom_received;
  bool ekf_accel_received;
  bool currently_over_covariance_limit;
  bool consecutive_turns_over_covariance_limit;
  bool recommends_failure;
} eufs_msgs__msg__EKFState;

// Struct for a sequence of eufs_msgs__msg__EKFState.
typedef struct eufs_msgs__msg__EKFState__Sequence
{
  eufs_msgs__msg__EKFState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__EKFState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__EKF_STATE__STRUCT_H_
