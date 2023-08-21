// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/PathIntegralStats.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/path_integral_stats__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `tag`
#include "rosidl_runtime_c/string_functions.h"
// Member `params`
#include "eufs_msgs/msg/detail/path_integral_params__functions.h"
// Member `stats`
#include "eufs_msgs/msg/detail/lap_stats__functions.h"

bool
eufs_msgs__msg__PathIntegralStats__init(eufs_msgs__msg__PathIntegralStats * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__PathIntegralStats__fini(msg);
    return false;
  }
  // tag
  if (!rosidl_runtime_c__String__init(&msg->tag)) {
    eufs_msgs__msg__PathIntegralStats__fini(msg);
    return false;
  }
  // params
  if (!eufs_msgs__msg__PathIntegralParams__init(&msg->params)) {
    eufs_msgs__msg__PathIntegralStats__fini(msg);
    return false;
  }
  // stats
  if (!eufs_msgs__msg__LapStats__init(&msg->stats)) {
    eufs_msgs__msg__PathIntegralStats__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__msg__PathIntegralStats__fini(eufs_msgs__msg__PathIntegralStats * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // tag
  rosidl_runtime_c__String__fini(&msg->tag);
  // params
  eufs_msgs__msg__PathIntegralParams__fini(&msg->params);
  // stats
  eufs_msgs__msg__LapStats__fini(&msg->stats);
}

bool
eufs_msgs__msg__PathIntegralStats__are_equal(const eufs_msgs__msg__PathIntegralStats * lhs, const eufs_msgs__msg__PathIntegralStats * rhs)
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
  // tag
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->tag), &(rhs->tag)))
  {
    return false;
  }
  // params
  if (!eufs_msgs__msg__PathIntegralParams__are_equal(
      &(lhs->params), &(rhs->params)))
  {
    return false;
  }
  // stats
  if (!eufs_msgs__msg__LapStats__are_equal(
      &(lhs->stats), &(rhs->stats)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__PathIntegralStats__copy(
  const eufs_msgs__msg__PathIntegralStats * input,
  eufs_msgs__msg__PathIntegralStats * output)
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
  // tag
  if (!rosidl_runtime_c__String__copy(
      &(input->tag), &(output->tag)))
  {
    return false;
  }
  // params
  if (!eufs_msgs__msg__PathIntegralParams__copy(
      &(input->params), &(output->params)))
  {
    return false;
  }
  // stats
  if (!eufs_msgs__msg__LapStats__copy(
      &(input->stats), &(output->stats)))
  {
    return false;
  }
  return true;
}

eufs_msgs__msg__PathIntegralStats *
eufs_msgs__msg__PathIntegralStats__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PathIntegralStats * msg = (eufs_msgs__msg__PathIntegralStats *)allocator.allocate(sizeof(eufs_msgs__msg__PathIntegralStats), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__PathIntegralStats));
  bool success = eufs_msgs__msg__PathIntegralStats__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__PathIntegralStats__destroy(eufs_msgs__msg__PathIntegralStats * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__PathIntegralStats__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__PathIntegralStats__Sequence__init(eufs_msgs__msg__PathIntegralStats__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PathIntegralStats * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__PathIntegralStats *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__PathIntegralStats), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__PathIntegralStats__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__PathIntegralStats__fini(&data[i - 1]);
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
eufs_msgs__msg__PathIntegralStats__Sequence__fini(eufs_msgs__msg__PathIntegralStats__Sequence * array)
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
      eufs_msgs__msg__PathIntegralStats__fini(&array->data[i]);
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

eufs_msgs__msg__PathIntegralStats__Sequence *
eufs_msgs__msg__PathIntegralStats__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PathIntegralStats__Sequence * array = (eufs_msgs__msg__PathIntegralStats__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__PathIntegralStats__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__PathIntegralStats__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__PathIntegralStats__Sequence__destroy(eufs_msgs__msg__PathIntegralStats__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__PathIntegralStats__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__PathIntegralStats__Sequence__are_equal(const eufs_msgs__msg__PathIntegralStats__Sequence * lhs, const eufs_msgs__msg__PathIntegralStats__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__PathIntegralStats__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__PathIntegralStats__Sequence__copy(
  const eufs_msgs__msg__PathIntegralStats__Sequence * input,
  eufs_msgs__msg__PathIntegralStats__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__PathIntegralStats);
    eufs_msgs__msg__PathIntegralStats * data =
      (eufs_msgs__msg__PathIntegralStats *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__PathIntegralStats__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__PathIntegralStats__fini(&data[i]);
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
    if (!eufs_msgs__msg__PathIntegralStats__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
