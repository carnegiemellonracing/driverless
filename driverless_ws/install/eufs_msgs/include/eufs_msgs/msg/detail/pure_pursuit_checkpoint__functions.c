// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/PurePursuitCheckpoint.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/pure_pursuit_checkpoint__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `position`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
eufs_msgs__msg__PurePursuitCheckpoint__init(eufs_msgs__msg__PurePursuitCheckpoint * msg)
{
  if (!msg) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__init(&msg->position)) {
    eufs_msgs__msg__PurePursuitCheckpoint__fini(msg);
    return false;
  }
  // max_speed
  // max_lateral_acceleration
  return true;
}

void
eufs_msgs__msg__PurePursuitCheckpoint__fini(eufs_msgs__msg__PurePursuitCheckpoint * msg)
{
  if (!msg) {
    return;
  }
  // position
  geometry_msgs__msg__Point__fini(&msg->position);
  // max_speed
  // max_lateral_acceleration
}

bool
eufs_msgs__msg__PurePursuitCheckpoint__are_equal(const eufs_msgs__msg__PurePursuitCheckpoint * lhs, const eufs_msgs__msg__PurePursuitCheckpoint * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__are_equal(
      &(lhs->position), &(rhs->position)))
  {
    return false;
  }
  // max_speed
  if (lhs->max_speed != rhs->max_speed) {
    return false;
  }
  // max_lateral_acceleration
  if (lhs->max_lateral_acceleration != rhs->max_lateral_acceleration) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__PurePursuitCheckpoint__copy(
  const eufs_msgs__msg__PurePursuitCheckpoint * input,
  eufs_msgs__msg__PurePursuitCheckpoint * output)
{
  if (!input || !output) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__copy(
      &(input->position), &(output->position)))
  {
    return false;
  }
  // max_speed
  output->max_speed = input->max_speed;
  // max_lateral_acceleration
  output->max_lateral_acceleration = input->max_lateral_acceleration;
  return true;
}

eufs_msgs__msg__PurePursuitCheckpoint *
eufs_msgs__msg__PurePursuitCheckpoint__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PurePursuitCheckpoint * msg = (eufs_msgs__msg__PurePursuitCheckpoint *)allocator.allocate(sizeof(eufs_msgs__msg__PurePursuitCheckpoint), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__PurePursuitCheckpoint));
  bool success = eufs_msgs__msg__PurePursuitCheckpoint__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__PurePursuitCheckpoint__destroy(eufs_msgs__msg__PurePursuitCheckpoint * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__PurePursuitCheckpoint__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__PurePursuitCheckpoint__Sequence__init(eufs_msgs__msg__PurePursuitCheckpoint__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PurePursuitCheckpoint * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__PurePursuitCheckpoint *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__PurePursuitCheckpoint), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__PurePursuitCheckpoint__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__PurePursuitCheckpoint__fini(&data[i - 1]);
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
eufs_msgs__msg__PurePursuitCheckpoint__Sequence__fini(eufs_msgs__msg__PurePursuitCheckpoint__Sequence * array)
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
      eufs_msgs__msg__PurePursuitCheckpoint__fini(&array->data[i]);
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

eufs_msgs__msg__PurePursuitCheckpoint__Sequence *
eufs_msgs__msg__PurePursuitCheckpoint__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PurePursuitCheckpoint__Sequence * array = (eufs_msgs__msg__PurePursuitCheckpoint__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__PurePursuitCheckpoint__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__PurePursuitCheckpoint__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__PurePursuitCheckpoint__Sequence__destroy(eufs_msgs__msg__PurePursuitCheckpoint__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__PurePursuitCheckpoint__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__PurePursuitCheckpoint__Sequence__are_equal(const eufs_msgs__msg__PurePursuitCheckpoint__Sequence * lhs, const eufs_msgs__msg__PurePursuitCheckpoint__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__PurePursuitCheckpoint__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__PurePursuitCheckpoint__Sequence__copy(
  const eufs_msgs__msg__PurePursuitCheckpoint__Sequence * input,
  eufs_msgs__msg__PurePursuitCheckpoint__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__PurePursuitCheckpoint);
    eufs_msgs__msg__PurePursuitCheckpoint * data =
      (eufs_msgs__msg__PurePursuitCheckpoint *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__PurePursuitCheckpoint__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__PurePursuitCheckpoint__fini(&data[i]);
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
    if (!eufs_msgs__msg__PurePursuitCheckpoint__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
