// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/IntegrationErr.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/integration_err__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
eufs_msgs__msg__IntegrationErr__init(eufs_msgs__msg__IntegrationErr * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__IntegrationErr__fini(msg);
    return false;
  }
  // position_err
  // orientation_err
  // linear_vel_err
  // angular_vel_err
  return true;
}

void
eufs_msgs__msg__IntegrationErr__fini(eufs_msgs__msg__IntegrationErr * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // position_err
  // orientation_err
  // linear_vel_err
  // angular_vel_err
}

bool
eufs_msgs__msg__IntegrationErr__are_equal(const eufs_msgs__msg__IntegrationErr * lhs, const eufs_msgs__msg__IntegrationErr * rhs)
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
  // position_err
  if (lhs->position_err != rhs->position_err) {
    return false;
  }
  // orientation_err
  if (lhs->orientation_err != rhs->orientation_err) {
    return false;
  }
  // linear_vel_err
  if (lhs->linear_vel_err != rhs->linear_vel_err) {
    return false;
  }
  // angular_vel_err
  if (lhs->angular_vel_err != rhs->angular_vel_err) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__IntegrationErr__copy(
  const eufs_msgs__msg__IntegrationErr * input,
  eufs_msgs__msg__IntegrationErr * output)
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
  // position_err
  output->position_err = input->position_err;
  // orientation_err
  output->orientation_err = input->orientation_err;
  // linear_vel_err
  output->linear_vel_err = input->linear_vel_err;
  // angular_vel_err
  output->angular_vel_err = input->angular_vel_err;
  return true;
}

eufs_msgs__msg__IntegrationErr *
eufs_msgs__msg__IntegrationErr__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__IntegrationErr * msg = (eufs_msgs__msg__IntegrationErr *)allocator.allocate(sizeof(eufs_msgs__msg__IntegrationErr), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__IntegrationErr));
  bool success = eufs_msgs__msg__IntegrationErr__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__IntegrationErr__destroy(eufs_msgs__msg__IntegrationErr * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__IntegrationErr__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__IntegrationErr__Sequence__init(eufs_msgs__msg__IntegrationErr__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__IntegrationErr * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__IntegrationErr *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__IntegrationErr), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__IntegrationErr__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__IntegrationErr__fini(&data[i - 1]);
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
eufs_msgs__msg__IntegrationErr__Sequence__fini(eufs_msgs__msg__IntegrationErr__Sequence * array)
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
      eufs_msgs__msg__IntegrationErr__fini(&array->data[i]);
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

eufs_msgs__msg__IntegrationErr__Sequence *
eufs_msgs__msg__IntegrationErr__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__IntegrationErr__Sequence * array = (eufs_msgs__msg__IntegrationErr__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__IntegrationErr__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__IntegrationErr__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__IntegrationErr__Sequence__destroy(eufs_msgs__msg__IntegrationErr__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__IntegrationErr__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__IntegrationErr__Sequence__are_equal(const eufs_msgs__msg__IntegrationErr__Sequence * lhs, const eufs_msgs__msg__IntegrationErr__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__IntegrationErr__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__IntegrationErr__Sequence__copy(
  const eufs_msgs__msg__IntegrationErr__Sequence * input,
  eufs_msgs__msg__IntegrationErr__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__IntegrationErr);
    eufs_msgs__msg__IntegrationErr * data =
      (eufs_msgs__msg__IntegrationErr *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__IntegrationErr__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__IntegrationErr__fini(&data[i]);
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
    if (!eufs_msgs__msg__IntegrationErr__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
