// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/CarState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/car_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `child_frame_id`
#include "rosidl_runtime_c/string_functions.h"
// Member `pose`
#include "geometry_msgs/msg/detail/pose_with_covariance__functions.h"
// Member `twist`
#include "geometry_msgs/msg/detail/twist_with_covariance__functions.h"
// Member `linear_acceleration`
#include "geometry_msgs/msg/detail/vector3__functions.h"

bool
eufs_msgs__msg__CarState__init(eufs_msgs__msg__CarState * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__CarState__fini(msg);
    return false;
  }
  // child_frame_id
  if (!rosidl_runtime_c__String__init(&msg->child_frame_id)) {
    eufs_msgs__msg__CarState__fini(msg);
    return false;
  }
  // pose
  if (!geometry_msgs__msg__PoseWithCovariance__init(&msg->pose)) {
    eufs_msgs__msg__CarState__fini(msg);
    return false;
  }
  // twist
  if (!geometry_msgs__msg__TwistWithCovariance__init(&msg->twist)) {
    eufs_msgs__msg__CarState__fini(msg);
    return false;
  }
  // linear_acceleration
  if (!geometry_msgs__msg__Vector3__init(&msg->linear_acceleration)) {
    eufs_msgs__msg__CarState__fini(msg);
    return false;
  }
  // linear_acceleration_covariance
  // slip_angle
  // state_of_charge
  return true;
}

void
eufs_msgs__msg__CarState__fini(eufs_msgs__msg__CarState * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // child_frame_id
  rosidl_runtime_c__String__fini(&msg->child_frame_id);
  // pose
  geometry_msgs__msg__PoseWithCovariance__fini(&msg->pose);
  // twist
  geometry_msgs__msg__TwistWithCovariance__fini(&msg->twist);
  // linear_acceleration
  geometry_msgs__msg__Vector3__fini(&msg->linear_acceleration);
  // linear_acceleration_covariance
  // slip_angle
  // state_of_charge
}

bool
eufs_msgs__msg__CarState__are_equal(const eufs_msgs__msg__CarState * lhs, const eufs_msgs__msg__CarState * rhs)
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
  // child_frame_id
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->child_frame_id), &(rhs->child_frame_id)))
  {
    return false;
  }
  // pose
  if (!geometry_msgs__msg__PoseWithCovariance__are_equal(
      &(lhs->pose), &(rhs->pose)))
  {
    return false;
  }
  // twist
  if (!geometry_msgs__msg__TwistWithCovariance__are_equal(
      &(lhs->twist), &(rhs->twist)))
  {
    return false;
  }
  // linear_acceleration
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->linear_acceleration), &(rhs->linear_acceleration)))
  {
    return false;
  }
  // linear_acceleration_covariance
  for (size_t i = 0; i < 9; ++i) {
    if (lhs->linear_acceleration_covariance[i] != rhs->linear_acceleration_covariance[i]) {
      return false;
    }
  }
  // slip_angle
  if (lhs->slip_angle != rhs->slip_angle) {
    return false;
  }
  // state_of_charge
  if (lhs->state_of_charge != rhs->state_of_charge) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__CarState__copy(
  const eufs_msgs__msg__CarState * input,
  eufs_msgs__msg__CarState * output)
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
  // child_frame_id
  if (!rosidl_runtime_c__String__copy(
      &(input->child_frame_id), &(output->child_frame_id)))
  {
    return false;
  }
  // pose
  if (!geometry_msgs__msg__PoseWithCovariance__copy(
      &(input->pose), &(output->pose)))
  {
    return false;
  }
  // twist
  if (!geometry_msgs__msg__TwistWithCovariance__copy(
      &(input->twist), &(output->twist)))
  {
    return false;
  }
  // linear_acceleration
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->linear_acceleration), &(output->linear_acceleration)))
  {
    return false;
  }
  // linear_acceleration_covariance
  for (size_t i = 0; i < 9; ++i) {
    output->linear_acceleration_covariance[i] = input->linear_acceleration_covariance[i];
  }
  // slip_angle
  output->slip_angle = input->slip_angle;
  // state_of_charge
  output->state_of_charge = input->state_of_charge;
  return true;
}

eufs_msgs__msg__CarState *
eufs_msgs__msg__CarState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__CarState * msg = (eufs_msgs__msg__CarState *)allocator.allocate(sizeof(eufs_msgs__msg__CarState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__CarState));
  bool success = eufs_msgs__msg__CarState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__CarState__destroy(eufs_msgs__msg__CarState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__CarState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__CarState__Sequence__init(eufs_msgs__msg__CarState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__CarState * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__CarState *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__CarState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__CarState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__CarState__fini(&data[i - 1]);
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
eufs_msgs__msg__CarState__Sequence__fini(eufs_msgs__msg__CarState__Sequence * array)
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
      eufs_msgs__msg__CarState__fini(&array->data[i]);
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

eufs_msgs__msg__CarState__Sequence *
eufs_msgs__msg__CarState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__CarState__Sequence * array = (eufs_msgs__msg__CarState__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__CarState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__CarState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__CarState__Sequence__destroy(eufs_msgs__msg__CarState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__CarState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__CarState__Sequence__are_equal(const eufs_msgs__msg__CarState__Sequence * lhs, const eufs_msgs__msg__CarState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__CarState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__CarState__Sequence__copy(
  const eufs_msgs__msg__CarState__Sequence * input,
  eufs_msgs__msg__CarState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__CarState);
    eufs_msgs__msg__CarState * data =
      (eufs_msgs__msg__CarState *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__CarState__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__CarState__fini(&data[i]);
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
    if (!eufs_msgs__msg__CarState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
