// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/Costmap.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/costmap__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `channel0`
// Member `channel1`
// Member `channel2`
// Member `channel3`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
eufs_msgs__msg__Costmap__init(eufs_msgs__msg__Costmap * msg)
{
  if (!msg) {
    return false;
  }
  // x_bounds_min
  // x_bounds_max
  // y_bounds_min
  // y_bounds_max
  // pixels_per_meter
  // channel0
  if (!rosidl_runtime_c__float__Sequence__init(&msg->channel0, 0)) {
    eufs_msgs__msg__Costmap__fini(msg);
    return false;
  }
  // channel1
  if (!rosidl_runtime_c__float__Sequence__init(&msg->channel1, 0)) {
    eufs_msgs__msg__Costmap__fini(msg);
    return false;
  }
  // channel2
  if (!rosidl_runtime_c__float__Sequence__init(&msg->channel2, 0)) {
    eufs_msgs__msg__Costmap__fini(msg);
    return false;
  }
  // channel3
  if (!rosidl_runtime_c__float__Sequence__init(&msg->channel3, 0)) {
    eufs_msgs__msg__Costmap__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__msg__Costmap__fini(eufs_msgs__msg__Costmap * msg)
{
  if (!msg) {
    return;
  }
  // x_bounds_min
  // x_bounds_max
  // y_bounds_min
  // y_bounds_max
  // pixels_per_meter
  // channel0
  rosidl_runtime_c__float__Sequence__fini(&msg->channel0);
  // channel1
  rosidl_runtime_c__float__Sequence__fini(&msg->channel1);
  // channel2
  rosidl_runtime_c__float__Sequence__fini(&msg->channel2);
  // channel3
  rosidl_runtime_c__float__Sequence__fini(&msg->channel3);
}

bool
eufs_msgs__msg__Costmap__are_equal(const eufs_msgs__msg__Costmap * lhs, const eufs_msgs__msg__Costmap * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // x_bounds_min
  if (lhs->x_bounds_min != rhs->x_bounds_min) {
    return false;
  }
  // x_bounds_max
  if (lhs->x_bounds_max != rhs->x_bounds_max) {
    return false;
  }
  // y_bounds_min
  if (lhs->y_bounds_min != rhs->y_bounds_min) {
    return false;
  }
  // y_bounds_max
  if (lhs->y_bounds_max != rhs->y_bounds_max) {
    return false;
  }
  // pixels_per_meter
  if (lhs->pixels_per_meter != rhs->pixels_per_meter) {
    return false;
  }
  // channel0
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->channel0), &(rhs->channel0)))
  {
    return false;
  }
  // channel1
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->channel1), &(rhs->channel1)))
  {
    return false;
  }
  // channel2
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->channel2), &(rhs->channel2)))
  {
    return false;
  }
  // channel3
  if (!rosidl_runtime_c__float__Sequence__are_equal(
      &(lhs->channel3), &(rhs->channel3)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__Costmap__copy(
  const eufs_msgs__msg__Costmap * input,
  eufs_msgs__msg__Costmap * output)
{
  if (!input || !output) {
    return false;
  }
  // x_bounds_min
  output->x_bounds_min = input->x_bounds_min;
  // x_bounds_max
  output->x_bounds_max = input->x_bounds_max;
  // y_bounds_min
  output->y_bounds_min = input->y_bounds_min;
  // y_bounds_max
  output->y_bounds_max = input->y_bounds_max;
  // pixels_per_meter
  output->pixels_per_meter = input->pixels_per_meter;
  // channel0
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->channel0), &(output->channel0)))
  {
    return false;
  }
  // channel1
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->channel1), &(output->channel1)))
  {
    return false;
  }
  // channel2
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->channel2), &(output->channel2)))
  {
    return false;
  }
  // channel3
  if (!rosidl_runtime_c__float__Sequence__copy(
      &(input->channel3), &(output->channel3)))
  {
    return false;
  }
  return true;
}

eufs_msgs__msg__Costmap *
eufs_msgs__msg__Costmap__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__Costmap * msg = (eufs_msgs__msg__Costmap *)allocator.allocate(sizeof(eufs_msgs__msg__Costmap), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__Costmap));
  bool success = eufs_msgs__msg__Costmap__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__Costmap__destroy(eufs_msgs__msg__Costmap * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__Costmap__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__Costmap__Sequence__init(eufs_msgs__msg__Costmap__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__Costmap * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__Costmap *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__Costmap), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__Costmap__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__Costmap__fini(&data[i - 1]);
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
eufs_msgs__msg__Costmap__Sequence__fini(eufs_msgs__msg__Costmap__Sequence * array)
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
      eufs_msgs__msg__Costmap__fini(&array->data[i]);
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

eufs_msgs__msg__Costmap__Sequence *
eufs_msgs__msg__Costmap__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__Costmap__Sequence * array = (eufs_msgs__msg__Costmap__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__Costmap__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__Costmap__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__Costmap__Sequence__destroy(eufs_msgs__msg__Costmap__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__Costmap__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__Costmap__Sequence__are_equal(const eufs_msgs__msg__Costmap__Sequence * lhs, const eufs_msgs__msg__Costmap__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__Costmap__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__Costmap__Sequence__copy(
  const eufs_msgs__msg__Costmap__Sequence * input,
  eufs_msgs__msg__Costmap__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__Costmap);
    eufs_msgs__msg__Costmap * data =
      (eufs_msgs__msg__Costmap *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__Costmap__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__Costmap__fini(&data[i]);
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
    if (!eufs_msgs__msg__Costmap__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
