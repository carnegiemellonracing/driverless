// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/PathIntegralParams.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/path_integral_params__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `map_path`
#include "rosidl_runtime_c/string_functions.h"

bool
eufs_msgs__msg__PathIntegralParams__init(eufs_msgs__msg__PathIntegralParams * msg)
{
  if (!msg) {
    return false;
  }
  // hz
  // num_timesteps
  // num_iters
  // gamma
  // init_steering
  // init_throttle
  // steering_var
  // throttle_var
  // max_throttle
  // speed_coefficient
  // track_coefficient
  // max_slip_angle
  // track_slop
  // crash_coeff
  // map_path
  if (!rosidl_runtime_c__String__init(&msg->map_path)) {
    eufs_msgs__msg__PathIntegralParams__fini(msg);
    return false;
  }
  // desired_speed
  return true;
}

void
eufs_msgs__msg__PathIntegralParams__fini(eufs_msgs__msg__PathIntegralParams * msg)
{
  if (!msg) {
    return;
  }
  // hz
  // num_timesteps
  // num_iters
  // gamma
  // init_steering
  // init_throttle
  // steering_var
  // throttle_var
  // max_throttle
  // speed_coefficient
  // track_coefficient
  // max_slip_angle
  // track_slop
  // crash_coeff
  // map_path
  rosidl_runtime_c__String__fini(&msg->map_path);
  // desired_speed
}

bool
eufs_msgs__msg__PathIntegralParams__are_equal(const eufs_msgs__msg__PathIntegralParams * lhs, const eufs_msgs__msg__PathIntegralParams * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // hz
  if (lhs->hz != rhs->hz) {
    return false;
  }
  // num_timesteps
  if (lhs->num_timesteps != rhs->num_timesteps) {
    return false;
  }
  // num_iters
  if (lhs->num_iters != rhs->num_iters) {
    return false;
  }
  // gamma
  if (lhs->gamma != rhs->gamma) {
    return false;
  }
  // init_steering
  if (lhs->init_steering != rhs->init_steering) {
    return false;
  }
  // init_throttle
  if (lhs->init_throttle != rhs->init_throttle) {
    return false;
  }
  // steering_var
  if (lhs->steering_var != rhs->steering_var) {
    return false;
  }
  // throttle_var
  if (lhs->throttle_var != rhs->throttle_var) {
    return false;
  }
  // max_throttle
  if (lhs->max_throttle != rhs->max_throttle) {
    return false;
  }
  // speed_coefficient
  if (lhs->speed_coefficient != rhs->speed_coefficient) {
    return false;
  }
  // track_coefficient
  if (lhs->track_coefficient != rhs->track_coefficient) {
    return false;
  }
  // max_slip_angle
  if (lhs->max_slip_angle != rhs->max_slip_angle) {
    return false;
  }
  // track_slop
  if (lhs->track_slop != rhs->track_slop) {
    return false;
  }
  // crash_coeff
  if (lhs->crash_coeff != rhs->crash_coeff) {
    return false;
  }
  // map_path
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->map_path), &(rhs->map_path)))
  {
    return false;
  }
  // desired_speed
  if (lhs->desired_speed != rhs->desired_speed) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__PathIntegralParams__copy(
  const eufs_msgs__msg__PathIntegralParams * input,
  eufs_msgs__msg__PathIntegralParams * output)
{
  if (!input || !output) {
    return false;
  }
  // hz
  output->hz = input->hz;
  // num_timesteps
  output->num_timesteps = input->num_timesteps;
  // num_iters
  output->num_iters = input->num_iters;
  // gamma
  output->gamma = input->gamma;
  // init_steering
  output->init_steering = input->init_steering;
  // init_throttle
  output->init_throttle = input->init_throttle;
  // steering_var
  output->steering_var = input->steering_var;
  // throttle_var
  output->throttle_var = input->throttle_var;
  // max_throttle
  output->max_throttle = input->max_throttle;
  // speed_coefficient
  output->speed_coefficient = input->speed_coefficient;
  // track_coefficient
  output->track_coefficient = input->track_coefficient;
  // max_slip_angle
  output->max_slip_angle = input->max_slip_angle;
  // track_slop
  output->track_slop = input->track_slop;
  // crash_coeff
  output->crash_coeff = input->crash_coeff;
  // map_path
  if (!rosidl_runtime_c__String__copy(
      &(input->map_path), &(output->map_path)))
  {
    return false;
  }
  // desired_speed
  output->desired_speed = input->desired_speed;
  return true;
}

eufs_msgs__msg__PathIntegralParams *
eufs_msgs__msg__PathIntegralParams__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PathIntegralParams * msg = (eufs_msgs__msg__PathIntegralParams *)allocator.allocate(sizeof(eufs_msgs__msg__PathIntegralParams), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__PathIntegralParams));
  bool success = eufs_msgs__msg__PathIntegralParams__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__PathIntegralParams__destroy(eufs_msgs__msg__PathIntegralParams * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__PathIntegralParams__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__PathIntegralParams__Sequence__init(eufs_msgs__msg__PathIntegralParams__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PathIntegralParams * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__PathIntegralParams *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__PathIntegralParams), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__PathIntegralParams__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__PathIntegralParams__fini(&data[i - 1]);
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
eufs_msgs__msg__PathIntegralParams__Sequence__fini(eufs_msgs__msg__PathIntegralParams__Sequence * array)
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
      eufs_msgs__msg__PathIntegralParams__fini(&array->data[i]);
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

eufs_msgs__msg__PathIntegralParams__Sequence *
eufs_msgs__msg__PathIntegralParams__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PathIntegralParams__Sequence * array = (eufs_msgs__msg__PathIntegralParams__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__PathIntegralParams__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__PathIntegralParams__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__PathIntegralParams__Sequence__destroy(eufs_msgs__msg__PathIntegralParams__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__PathIntegralParams__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__PathIntegralParams__Sequence__are_equal(const eufs_msgs__msg__PathIntegralParams__Sequence * lhs, const eufs_msgs__msg__PathIntegralParams__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__PathIntegralParams__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__PathIntegralParams__Sequence__copy(
  const eufs_msgs__msg__PathIntegralParams__Sequence * input,
  eufs_msgs__msg__PathIntegralParams__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__PathIntegralParams);
    eufs_msgs__msg__PathIntegralParams * data =
      (eufs_msgs__msg__PathIntegralParams *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__PathIntegralParams__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__PathIntegralParams__fini(&data[i]);
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
    if (!eufs_msgs__msg__PathIntegralParams__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
