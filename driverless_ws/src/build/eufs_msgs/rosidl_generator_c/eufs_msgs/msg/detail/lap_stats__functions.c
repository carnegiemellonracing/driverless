// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/LapStats.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/lap_stats__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
eufs_msgs__msg__LapStats__init(eufs_msgs__msg__LapStats * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__LapStats__fini(msg);
    return false;
  }
  // lap_number
  // lap_time
  // avg_speed
  // max_speed
  // speed_var
  // max_slip
  // max_lateral_accel
  // normalized_deviation_mse
  // deviation_var
  // max_deviation
  return true;
}

void
eufs_msgs__msg__LapStats__fini(eufs_msgs__msg__LapStats * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // lap_number
  // lap_time
  // avg_speed
  // max_speed
  // speed_var
  // max_slip
  // max_lateral_accel
  // normalized_deviation_mse
  // deviation_var
  // max_deviation
}

bool
eufs_msgs__msg__LapStats__are_equal(const eufs_msgs__msg__LapStats * lhs, const eufs_msgs__msg__LapStats * rhs)
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
  // lap_number
  if (lhs->lap_number != rhs->lap_number) {
    return false;
  }
  // lap_time
  if (lhs->lap_time != rhs->lap_time) {
    return false;
  }
  // avg_speed
  if (lhs->avg_speed != rhs->avg_speed) {
    return false;
  }
  // max_speed
  if (lhs->max_speed != rhs->max_speed) {
    return false;
  }
  // speed_var
  if (lhs->speed_var != rhs->speed_var) {
    return false;
  }
  // max_slip
  if (lhs->max_slip != rhs->max_slip) {
    return false;
  }
  // max_lateral_accel
  if (lhs->max_lateral_accel != rhs->max_lateral_accel) {
    return false;
  }
  // normalized_deviation_mse
  if (lhs->normalized_deviation_mse != rhs->normalized_deviation_mse) {
    return false;
  }
  // deviation_var
  if (lhs->deviation_var != rhs->deviation_var) {
    return false;
  }
  // max_deviation
  if (lhs->max_deviation != rhs->max_deviation) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__LapStats__copy(
  const eufs_msgs__msg__LapStats * input,
  eufs_msgs__msg__LapStats * output)
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
  // lap_number
  output->lap_number = input->lap_number;
  // lap_time
  output->lap_time = input->lap_time;
  // avg_speed
  output->avg_speed = input->avg_speed;
  // max_speed
  output->max_speed = input->max_speed;
  // speed_var
  output->speed_var = input->speed_var;
  // max_slip
  output->max_slip = input->max_slip;
  // max_lateral_accel
  output->max_lateral_accel = input->max_lateral_accel;
  // normalized_deviation_mse
  output->normalized_deviation_mse = input->normalized_deviation_mse;
  // deviation_var
  output->deviation_var = input->deviation_var;
  // max_deviation
  output->max_deviation = input->max_deviation;
  return true;
}

eufs_msgs__msg__LapStats *
eufs_msgs__msg__LapStats__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__LapStats * msg = (eufs_msgs__msg__LapStats *)allocator.allocate(sizeof(eufs_msgs__msg__LapStats), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__LapStats));
  bool success = eufs_msgs__msg__LapStats__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__LapStats__destroy(eufs_msgs__msg__LapStats * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__LapStats__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__LapStats__Sequence__init(eufs_msgs__msg__LapStats__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__LapStats * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__LapStats *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__LapStats), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__LapStats__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__LapStats__fini(&data[i - 1]);
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
eufs_msgs__msg__LapStats__Sequence__fini(eufs_msgs__msg__LapStats__Sequence * array)
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
      eufs_msgs__msg__LapStats__fini(&array->data[i]);
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

eufs_msgs__msg__LapStats__Sequence *
eufs_msgs__msg__LapStats__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__LapStats__Sequence * array = (eufs_msgs__msg__LapStats__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__LapStats__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__LapStats__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__LapStats__Sequence__destroy(eufs_msgs__msg__LapStats__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__LapStats__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__LapStats__Sequence__are_equal(const eufs_msgs__msg__LapStats__Sequence * lhs, const eufs_msgs__msg__LapStats__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__LapStats__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__LapStats__Sequence__copy(
  const eufs_msgs__msg__LapStats__Sequence * input,
  eufs_msgs__msg__LapStats__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__LapStats);
    eufs_msgs__msg__LapStats * data =
      (eufs_msgs__msg__LapStats *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__LapStats__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__LapStats__fini(&data[i]);
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
    if (!eufs_msgs__msg__LapStats__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
