// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/EKFErr.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/ekf_err__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
eufs_msgs__msg__EKFErr__init(eufs_msgs__msg__EKFErr * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__EKFErr__fini(msg);
    return false;
  }
  // gps_x_vel_err
  // gps_y_vel_err
  // imu_x_acc_err
  // imu_y_acc_err
  // imu_yaw_err
  // ekf_x_vel_var
  // ekf_y_vel_var
  // ekf_x_acc_var
  // ekf_y_acc_var
  // ekf_yaw_var
  return true;
}

void
eufs_msgs__msg__EKFErr__fini(eufs_msgs__msg__EKFErr * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // gps_x_vel_err
  // gps_y_vel_err
  // imu_x_acc_err
  // imu_y_acc_err
  // imu_yaw_err
  // ekf_x_vel_var
  // ekf_y_vel_var
  // ekf_x_acc_var
  // ekf_y_acc_var
  // ekf_yaw_var
}

bool
eufs_msgs__msg__EKFErr__are_equal(const eufs_msgs__msg__EKFErr * lhs, const eufs_msgs__msg__EKFErr * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // gps_x_vel_err
  if (lhs->gps_x_vel_err != rhs->gps_x_vel_err) {
    return false;
  }
  // gps_y_vel_err
  if (lhs->gps_y_vel_err != rhs->gps_y_vel_err) {
    return false;
  }
  // imu_x_acc_err
  if (lhs->imu_x_acc_err != rhs->imu_x_acc_err) {
    return false;
  }
  // imu_y_acc_err
  if (lhs->imu_y_acc_err != rhs->imu_y_acc_err) {
    return false;
  }
  // imu_yaw_err
  if (lhs->imu_yaw_err != rhs->imu_yaw_err) {
    return false;
  }
  // ekf_x_vel_var
  if (lhs->ekf_x_vel_var != rhs->ekf_x_vel_var) {
    return false;
  }
  // ekf_y_vel_var
  if (lhs->ekf_y_vel_var != rhs->ekf_y_vel_var) {
    return false;
  }
  // ekf_x_acc_var
  if (lhs->ekf_x_acc_var != rhs->ekf_x_acc_var) {
    return false;
  }
  // ekf_y_acc_var
  if (lhs->ekf_y_acc_var != rhs->ekf_y_acc_var) {
    return false;
  }
  // ekf_yaw_var
  if (lhs->ekf_yaw_var != rhs->ekf_yaw_var) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__EKFErr__copy(
  const eufs_msgs__msg__EKFErr * input,
  eufs_msgs__msg__EKFErr * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // gps_x_vel_err
  output->gps_x_vel_err = input->gps_x_vel_err;
  // gps_y_vel_err
  output->gps_y_vel_err = input->gps_y_vel_err;
  // imu_x_acc_err
  output->imu_x_acc_err = input->imu_x_acc_err;
  // imu_y_acc_err
  output->imu_y_acc_err = input->imu_y_acc_err;
  // imu_yaw_err
  output->imu_yaw_err = input->imu_yaw_err;
  // ekf_x_vel_var
  output->ekf_x_vel_var = input->ekf_x_vel_var;
  // ekf_y_vel_var
  output->ekf_y_vel_var = input->ekf_y_vel_var;
  // ekf_x_acc_var
  output->ekf_x_acc_var = input->ekf_x_acc_var;
  // ekf_y_acc_var
  output->ekf_y_acc_var = input->ekf_y_acc_var;
  // ekf_yaw_var
  output->ekf_yaw_var = input->ekf_yaw_var;
  return true;
}

eufs_msgs__msg__EKFErr *
eufs_msgs__msg__EKFErr__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__EKFErr * msg = (eufs_msgs__msg__EKFErr *)allocator.allocate(sizeof(eufs_msgs__msg__EKFErr), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__EKFErr));
  bool success = eufs_msgs__msg__EKFErr__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__EKFErr__destroy(eufs_msgs__msg__EKFErr * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__EKFErr__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__EKFErr__Sequence__init(eufs_msgs__msg__EKFErr__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__EKFErr * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__EKFErr *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__EKFErr), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__EKFErr__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__EKFErr__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
eufs_msgs__msg__EKFErr__Sequence__fini(eufs_msgs__msg__EKFErr__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      eufs_msgs__msg__EKFErr__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

eufs_msgs__msg__EKFErr__Sequence *
eufs_msgs__msg__EKFErr__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__EKFErr__Sequence * array = (eufs_msgs__msg__EKFErr__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__EKFErr__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__EKFErr__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__EKFErr__Sequence__destroy(eufs_msgs__msg__EKFErr__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__EKFErr__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__EKFErr__Sequence__are_equal(const eufs_msgs__msg__EKFErr__Sequence * lhs, const eufs_msgs__msg__EKFErr__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__EKFErr__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__EKFErr__Sequence__copy(
  const eufs_msgs__msg__EKFErr__Sequence * input,
  eufs_msgs__msg__EKFErr__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__EKFErr);
    eufs_msgs__msg__EKFErr * data =
      (eufs_msgs__msg__EKFErr *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__EKFErr__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__EKFErr__fini(&data[i]);
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
    if (!eufs_msgs__msg__EKFErr__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
