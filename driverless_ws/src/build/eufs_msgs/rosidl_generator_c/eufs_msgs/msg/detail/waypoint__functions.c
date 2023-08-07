// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/Waypoint.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/waypoint__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `position`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
eufs_msgs__msg__Waypoint__init(eufs_msgs__msg__Waypoint * msg)
{
  if (!msg) {
    return false;
  }
  // position
  if (!geometry_msgs__msg__Point__init(&msg->position)) {
    eufs_msgs__msg__Waypoint__fini(msg);
    return false;
  }
  // speed
  // suggested_steering
  return true;
}

void
eufs_msgs__msg__Waypoint__fini(eufs_msgs__msg__Waypoint * msg)
{
  if (!msg) {
    return;
  }
  // position
  geometry_msgs__msg__Point__fini(&msg->position);
  // speed
  // suggested_steering
}

bool
eufs_msgs__msg__Waypoint__are_equal(const eufs_msgs__msg__Waypoint * lhs, const eufs_msgs__msg__Waypoint * rhs)
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
  // speed
  if (lhs->speed != rhs->speed) {
    return false;
  }
  // suggested_steering
  if (lhs->suggested_steering != rhs->suggested_steering) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__Waypoint__copy(
  const eufs_msgs__msg__Waypoint * input,
  eufs_msgs__msg__Waypoint * output)
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
  // speed
  output->speed = input->speed;
  // suggested_steering
  output->suggested_steering = input->suggested_steering;
  return true;
}

eufs_msgs__msg__Waypoint *
eufs_msgs__msg__Waypoint__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__Waypoint * msg = (eufs_msgs__msg__Waypoint *)allocator.allocate(sizeof(eufs_msgs__msg__Waypoint), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__Waypoint));
  bool success = eufs_msgs__msg__Waypoint__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__Waypoint__destroy(eufs_msgs__msg__Waypoint * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__Waypoint__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__Waypoint__Sequence__init(eufs_msgs__msg__Waypoint__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__Waypoint * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__Waypoint *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__Waypoint), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__Waypoint__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__Waypoint__fini(&data[i - 1]);
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
eufs_msgs__msg__Waypoint__Sequence__fini(eufs_msgs__msg__Waypoint__Sequence * array)
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
      eufs_msgs__msg__Waypoint__fini(&array->data[i]);
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

eufs_msgs__msg__Waypoint__Sequence *
eufs_msgs__msg__Waypoint__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__Waypoint__Sequence * array = (eufs_msgs__msg__Waypoint__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__Waypoint__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__Waypoint__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__Waypoint__Sequence__destroy(eufs_msgs__msg__Waypoint__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__Waypoint__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__Waypoint__Sequence__are_equal(const eufs_msgs__msg__Waypoint__Sequence * lhs, const eufs_msgs__msg__Waypoint__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__Waypoint__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__Waypoint__Sequence__copy(
  const eufs_msgs__msg__Waypoint__Sequence * input,
  eufs_msgs__msg__Waypoint__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__Waypoint);
    eufs_msgs__msg__Waypoint * data =
      (eufs_msgs__msg__Waypoint *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__Waypoint__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__Waypoint__fini(&data[i]);
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
    if (!eufs_msgs__msg__Waypoint__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
