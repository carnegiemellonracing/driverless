// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/EKFState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/ekf_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>


bool
eufs_msgs__msg__EKFState__init(eufs_msgs__msg__EKFState * msg)
{
  if (!msg) {
    return false;
  }
  // gps_received
  // imu_received
  // wheel_odom_received
  // ekf_odom_received
  // ekf_accel_received
  // currently_over_covariance_limit
  // consecutive_turns_over_covariance_limit
  // recommends_failure
  return true;
}

void
eufs_msgs__msg__EKFState__fini(eufs_msgs__msg__EKFState * msg)
{
  if (!msg) {
    return;
  }
  // gps_received
  // imu_received
  // wheel_odom_received
  // ekf_odom_received
  // ekf_accel_received
  // currently_over_covariance_limit
  // consecutive_turns_over_covariance_limit
  // recommends_failure
}

bool
eufs_msgs__msg__EKFState__are_equal(const eufs_msgs__msg__EKFState * lhs, const eufs_msgs__msg__EKFState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // gps_received
  if (lhs->gps_received != rhs->gps_received) {
    return false;
  }
  // imu_received
  if (lhs->imu_received != rhs->imu_received) {
    return false;
  }
  // wheel_odom_received
  if (lhs->wheel_odom_received != rhs->wheel_odom_received) {
    return false;
  }
  // ekf_odom_received
  if (lhs->ekf_odom_received != rhs->ekf_odom_received) {
    return false;
  }
  // ekf_accel_received
  if (lhs->ekf_accel_received != rhs->ekf_accel_received) {
    return false;
  }
  // currently_over_covariance_limit
  if (lhs->currently_over_covariance_limit != rhs->currently_over_covariance_limit) {
    return false;
  }
  // consecutive_turns_over_covariance_limit
  if (lhs->consecutive_turns_over_covariance_limit != rhs->consecutive_turns_over_covariance_limit) {
    return false;
  }
  // recommends_failure
  if (lhs->recommends_failure != rhs->recommends_failure) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__EKFState__copy(
  const eufs_msgs__msg__EKFState * input,
  eufs_msgs__msg__EKFState * output)
{
  if (!input || !output) {
    return false;
  }
  // gps_received
  output->gps_received = input->gps_received;
  // imu_received
  output->imu_received = input->imu_received;
  // wheel_odom_received
  output->wheel_odom_received = input->wheel_odom_received;
  // ekf_odom_received
  output->ekf_odom_received = input->ekf_odom_received;
  // ekf_accel_received
  output->ekf_accel_received = input->ekf_accel_received;
  // currently_over_covariance_limit
  output->currently_over_covariance_limit = input->currently_over_covariance_limit;
  // consecutive_turns_over_covariance_limit
  output->consecutive_turns_over_covariance_limit = input->consecutive_turns_over_covariance_limit;
  // recommends_failure
  output->recommends_failure = input->recommends_failure;
  return true;
}

eufs_msgs__msg__EKFState *
eufs_msgs__msg__EKFState__create()
{
  eufs_msgs__msg__EKFState * msg = (eufs_msgs__msg__EKFState *)malloc(sizeof(eufs_msgs__msg__EKFState));
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__EKFState));
  bool success = eufs_msgs__msg__EKFState__init(msg);
  if (!success) {
    free(msg);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__EKFState__destroy(eufs_msgs__msg__EKFState * msg)
{
  if (msg) {
    eufs_msgs__msg__EKFState__fini(msg);
  }
  free(msg);
}


bool
eufs_msgs__msg__EKFState__Sequence__init(eufs_msgs__msg__EKFState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  eufs_msgs__msg__EKFState * data = NULL;
  if (size) {
    data = (eufs_msgs__msg__EKFState *)calloc(size, sizeof(eufs_msgs__msg__EKFState));
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__EKFState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__EKFState__fini(&data[i - 1]);
      }
      free(data);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
eufs_msgs__msg__EKFState__Sequence__fini(eufs_msgs__msg__EKFState__Sequence * array)
{
  if (!array) {
    return;
  }
  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      eufs_msgs__msg__EKFState__fini(&array->data[i]);
    }
    free(array->data);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

eufs_msgs__msg__EKFState__Sequence *
eufs_msgs__msg__EKFState__Sequence__create(size_t size)
{
  eufs_msgs__msg__EKFState__Sequence * array = (eufs_msgs__msg__EKFState__Sequence *)malloc(sizeof(eufs_msgs__msg__EKFState__Sequence));
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__EKFState__Sequence__init(array, size);
  if (!success) {
    free(array);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__EKFState__Sequence__destroy(eufs_msgs__msg__EKFState__Sequence * array)
{
  if (array) {
    eufs_msgs__msg__EKFState__Sequence__fini(array);
  }
  free(array);
}

bool
eufs_msgs__msg__EKFState__Sequence__are_equal(const eufs_msgs__msg__EKFState__Sequence * lhs, const eufs_msgs__msg__EKFState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__EKFState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__EKFState__Sequence__copy(
  const eufs_msgs__msg__EKFState__Sequence * input,
  eufs_msgs__msg__EKFState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__EKFState);
    eufs_msgs__msg__EKFState * data =
      (eufs_msgs__msg__EKFState *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__EKFState__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__EKFState__fini(&data[i]);
        }
        free(data);
        return false;
      }
    }
    output->data = data;
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!eufs_msgs__msg__EKFState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
