// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/PointArrayStamped.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/point_array_stamped__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `points`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
eufs_msgs__msg__PointArrayStamped__init(eufs_msgs__msg__PointArrayStamped * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__PointArrayStamped__fini(msg);
    return false;
  }
  // points
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->points, 0)) {
    eufs_msgs__msg__PointArrayStamped__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__msg__PointArrayStamped__fini(eufs_msgs__msg__PointArrayStamped * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // points
  geometry_msgs__msg__Point__Sequence__fini(&msg->points);
}

bool
eufs_msgs__msg__PointArrayStamped__are_equal(const eufs_msgs__msg__PointArrayStamped * lhs, const eufs_msgs__msg__PointArrayStamped * rhs)
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
  // points
  if (!geometry_msgs__msg__Point__Sequence__are_equal(
      &(lhs->points), &(rhs->points)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__PointArrayStamped__copy(
  const eufs_msgs__msg__PointArrayStamped * input,
  eufs_msgs__msg__PointArrayStamped * output)
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
  // points
  if (!geometry_msgs__msg__Point__Sequence__copy(
      &(input->points), &(output->points)))
  {
    return false;
  }
  return true;
}

eufs_msgs__msg__PointArrayStamped *
eufs_msgs__msg__PointArrayStamped__create()
{
  eufs_msgs__msg__PointArrayStamped * msg = (eufs_msgs__msg__PointArrayStamped *)malloc(sizeof(eufs_msgs__msg__PointArrayStamped));
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__PointArrayStamped));
  bool success = eufs_msgs__msg__PointArrayStamped__init(msg);
  if (!success) {
    free(msg);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__PointArrayStamped__destroy(eufs_msgs__msg__PointArrayStamped * msg)
{
  if (msg) {
    eufs_msgs__msg__PointArrayStamped__fini(msg);
  }
  free(msg);
}


bool
eufs_msgs__msg__PointArrayStamped__Sequence__init(eufs_msgs__msg__PointArrayStamped__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  eufs_msgs__msg__PointArrayStamped * data = NULL;
  if (size) {
    data = (eufs_msgs__msg__PointArrayStamped *)calloc(size, sizeof(eufs_msgs__msg__PointArrayStamped));
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__PointArrayStamped__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__PointArrayStamped__fini(&data[i - 1]);
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
eufs_msgs__msg__PointArrayStamped__Sequence__fini(eufs_msgs__msg__PointArrayStamped__Sequence * array)
{
  if (!array) {
    return;
  }
  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      eufs_msgs__msg__PointArrayStamped__fini(&array->data[i]);
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

eufs_msgs__msg__PointArrayStamped__Sequence *
eufs_msgs__msg__PointArrayStamped__Sequence__create(size_t size)
{
  eufs_msgs__msg__PointArrayStamped__Sequence * array = (eufs_msgs__msg__PointArrayStamped__Sequence *)malloc(sizeof(eufs_msgs__msg__PointArrayStamped__Sequence));
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__PointArrayStamped__Sequence__init(array, size);
  if (!success) {
    free(array);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__PointArrayStamped__Sequence__destroy(eufs_msgs__msg__PointArrayStamped__Sequence * array)
{
  if (array) {
    eufs_msgs__msg__PointArrayStamped__Sequence__fini(array);
  }
  free(array);
}

bool
eufs_msgs__msg__PointArrayStamped__Sequence__are_equal(const eufs_msgs__msg__PointArrayStamped__Sequence * lhs, const eufs_msgs__msg__PointArrayStamped__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__PointArrayStamped__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__PointArrayStamped__Sequence__copy(
  const eufs_msgs__msg__PointArrayStamped__Sequence * input,
  eufs_msgs__msg__PointArrayStamped__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__PointArrayStamped);
    eufs_msgs__msg__PointArrayStamped * data =
      (eufs_msgs__msg__PointArrayStamped *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__PointArrayStamped__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__PointArrayStamped__fini(&data[i]);
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
    if (!eufs_msgs__msg__PointArrayStamped__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
