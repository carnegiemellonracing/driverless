// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cmrdv_interfaces:msg/DataFrame.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/data_frame__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `zed_left_img`
#include "sensor_msgs/msg/detail/image__functions.h"
// Member `zed_pts`
#include "sensor_msgs/msg/detail/point_cloud2__functions.h"
// Member `sbg`
#include "sbg_driver/msg/detail/sbg_gps_pos__functions.h"
// Member `imu`
#include "sbg_driver/msg/detail/sbg_imu_data__functions.h"

bool
cmrdv_interfaces__msg__DataFrame__init(cmrdv_interfaces__msg__DataFrame * msg)
{
  if (!msg) {
    return false;
  }
  // zed_left_img
  if (!sensor_msgs__msg__Image__init(&msg->zed_left_img)) {
    cmrdv_interfaces__msg__DataFrame__fini(msg);
    return false;
  }
  // zed_pts
  if (!sensor_msgs__msg__PointCloud2__init(&msg->zed_pts)) {
    cmrdv_interfaces__msg__DataFrame__fini(msg);
    return false;
  }
  // sbg
  if (!sbg_driver__msg__SbgGpsPos__init(&msg->sbg)) {
    cmrdv_interfaces__msg__DataFrame__fini(msg);
    return false;
  }
  // imu
  if (!sbg_driver__msg__SbgImuData__init(&msg->imu)) {
    cmrdv_interfaces__msg__DataFrame__fini(msg);
    return false;
  }
  return true;
}

void
cmrdv_interfaces__msg__DataFrame__fini(cmrdv_interfaces__msg__DataFrame * msg)
{
  if (!msg) {
    return;
  }
  // zed_left_img
  sensor_msgs__msg__Image__fini(&msg->zed_left_img);
  // zed_pts
  sensor_msgs__msg__PointCloud2__fini(&msg->zed_pts);
  // sbg
  sbg_driver__msg__SbgGpsPos__fini(&msg->sbg);
  // imu
  sbg_driver__msg__SbgImuData__fini(&msg->imu);
}

bool
cmrdv_interfaces__msg__DataFrame__are_equal(const cmrdv_interfaces__msg__DataFrame * lhs, const cmrdv_interfaces__msg__DataFrame * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // zed_left_img
  if (!sensor_msgs__msg__Image__are_equal(
      &(lhs->zed_left_img), &(rhs->zed_left_img)))
  {
    return false;
  }
  // zed_pts
  if (!sensor_msgs__msg__PointCloud2__are_equal(
      &(lhs->zed_pts), &(rhs->zed_pts)))
  {
    return false;
  }
  // sbg
  if (!sbg_driver__msg__SbgGpsPos__are_equal(
      &(lhs->sbg), &(rhs->sbg)))
  {
    return false;
  }
  // imu
  if (!sbg_driver__msg__SbgImuData__are_equal(
      &(lhs->imu), &(rhs->imu)))
  {
    return false;
  }
  return true;
}

bool
cmrdv_interfaces__msg__DataFrame__copy(
  const cmrdv_interfaces__msg__DataFrame * input,
  cmrdv_interfaces__msg__DataFrame * output)
{
  if (!input || !output) {
    return false;
  }
  // zed_left_img
  if (!sensor_msgs__msg__Image__copy(
      &(input->zed_left_img), &(output->zed_left_img)))
  {
    return false;
  }
  // zed_pts
  if (!sensor_msgs__msg__PointCloud2__copy(
      &(input->zed_pts), &(output->zed_pts)))
  {
    return false;
  }
  // sbg
  if (!sbg_driver__msg__SbgGpsPos__copy(
      &(input->sbg), &(output->sbg)))
  {
    return false;
  }
  // imu
  if (!sbg_driver__msg__SbgImuData__copy(
      &(input->imu), &(output->imu)))
  {
    return false;
  }
  return true;
}

cmrdv_interfaces__msg__DataFrame *
cmrdv_interfaces__msg__DataFrame__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__DataFrame * msg = (cmrdv_interfaces__msg__DataFrame *)allocator.allocate(sizeof(cmrdv_interfaces__msg__DataFrame), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cmrdv_interfaces__msg__DataFrame));
  bool success = cmrdv_interfaces__msg__DataFrame__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cmrdv_interfaces__msg__DataFrame__destroy(cmrdv_interfaces__msg__DataFrame * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cmrdv_interfaces__msg__DataFrame__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cmrdv_interfaces__msg__DataFrame__Sequence__init(cmrdv_interfaces__msg__DataFrame__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__DataFrame * data = NULL;

  if (size) {
    data = (cmrdv_interfaces__msg__DataFrame *)allocator.zero_allocate(size, sizeof(cmrdv_interfaces__msg__DataFrame), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cmrdv_interfaces__msg__DataFrame__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cmrdv_interfaces__msg__DataFrame__fini(&data[i - 1]);
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
cmrdv_interfaces__msg__DataFrame__Sequence__fini(cmrdv_interfaces__msg__DataFrame__Sequence * array)
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
      cmrdv_interfaces__msg__DataFrame__fini(&array->data[i]);
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

cmrdv_interfaces__msg__DataFrame__Sequence *
cmrdv_interfaces__msg__DataFrame__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__DataFrame__Sequence * array = (cmrdv_interfaces__msg__DataFrame__Sequence *)allocator.allocate(sizeof(cmrdv_interfaces__msg__DataFrame__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cmrdv_interfaces__msg__DataFrame__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cmrdv_interfaces__msg__DataFrame__Sequence__destroy(cmrdv_interfaces__msg__DataFrame__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cmrdv_interfaces__msg__DataFrame__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cmrdv_interfaces__msg__DataFrame__Sequence__are_equal(const cmrdv_interfaces__msg__DataFrame__Sequence * lhs, const cmrdv_interfaces__msg__DataFrame__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cmrdv_interfaces__msg__DataFrame__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cmrdv_interfaces__msg__DataFrame__Sequence__copy(
  const cmrdv_interfaces__msg__DataFrame__Sequence * input,
  cmrdv_interfaces__msg__DataFrame__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cmrdv_interfaces__msg__DataFrame);
    cmrdv_interfaces__msg__DataFrame * data =
      (cmrdv_interfaces__msg__DataFrame *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cmrdv_interfaces__msg__DataFrame__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          cmrdv_interfaces__msg__DataFrame__fini(&data[i]);
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
    if (!cmrdv_interfaces__msg__DataFrame__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
