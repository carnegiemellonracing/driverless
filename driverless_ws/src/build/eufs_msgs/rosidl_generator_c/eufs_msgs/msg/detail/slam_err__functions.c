// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/SLAMErr.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/slam_err__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
eufs_msgs__msg__SLAMErr__init(eufs_msgs__msg__SLAMErr * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__SLAMErr__fini(msg);
    return false;
  }
  // x_err
  // y_err
  // z_err
  // x_orient_err
  // y_orient_err
  // z_orient_err
  // w_orient_err
  // map_similarity
  return true;
}

void
eufs_msgs__msg__SLAMErr__fini(eufs_msgs__msg__SLAMErr * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // x_err
  // y_err
  // z_err
  // x_orient_err
  // y_orient_err
  // z_orient_err
  // w_orient_err
  // map_similarity
}

bool
eufs_msgs__msg__SLAMErr__are_equal(const eufs_msgs__msg__SLAMErr * lhs, const eufs_msgs__msg__SLAMErr * rhs)
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
  // x_err
  if (lhs->x_err != rhs->x_err) {
    return false;
  }
  // y_err
  if (lhs->y_err != rhs->y_err) {
    return false;
  }
  // z_err
  if (lhs->z_err != rhs->z_err) {
    return false;
  }
  // x_orient_err
  if (lhs->x_orient_err != rhs->x_orient_err) {
    return false;
  }
  // y_orient_err
  if (lhs->y_orient_err != rhs->y_orient_err) {
    return false;
  }
  // z_orient_err
  if (lhs->z_orient_err != rhs->z_orient_err) {
    return false;
  }
  // w_orient_err
  if (lhs->w_orient_err != rhs->w_orient_err) {
    return false;
  }
  // map_similarity
  if (lhs->map_similarity != rhs->map_similarity) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__SLAMErr__copy(
  const eufs_msgs__msg__SLAMErr * input,
  eufs_msgs__msg__SLAMErr * output)
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
  // x_err
  output->x_err = input->x_err;
  // y_err
  output->y_err = input->y_err;
  // z_err
  output->z_err = input->z_err;
  // x_orient_err
  output->x_orient_err = input->x_orient_err;
  // y_orient_err
  output->y_orient_err = input->y_orient_err;
  // z_orient_err
  output->z_orient_err = input->z_orient_err;
  // w_orient_err
  output->w_orient_err = input->w_orient_err;
  // map_similarity
  output->map_similarity = input->map_similarity;
  return true;
}

eufs_msgs__msg__SLAMErr *
eufs_msgs__msg__SLAMErr__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__SLAMErr * msg = (eufs_msgs__msg__SLAMErr *)allocator.allocate(sizeof(eufs_msgs__msg__SLAMErr), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__SLAMErr));
  bool success = eufs_msgs__msg__SLAMErr__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__SLAMErr__destroy(eufs_msgs__msg__SLAMErr * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__SLAMErr__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__SLAMErr__Sequence__init(eufs_msgs__msg__SLAMErr__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__SLAMErr * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__SLAMErr *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__SLAMErr), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__SLAMErr__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__SLAMErr__fini(&data[i - 1]);
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
eufs_msgs__msg__SLAMErr__Sequence__fini(eufs_msgs__msg__SLAMErr__Sequence * array)
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
      eufs_msgs__msg__SLAMErr__fini(&array->data[i]);
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

eufs_msgs__msg__SLAMErr__Sequence *
eufs_msgs__msg__SLAMErr__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__SLAMErr__Sequence * array = (eufs_msgs__msg__SLAMErr__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__SLAMErr__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__SLAMErr__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__SLAMErr__Sequence__destroy(eufs_msgs__msg__SLAMErr__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__SLAMErr__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__SLAMErr__Sequence__are_equal(const eufs_msgs__msg__SLAMErr__Sequence * lhs, const eufs_msgs__msg__SLAMErr__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__SLAMErr__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__SLAMErr__Sequence__copy(
  const eufs_msgs__msg__SLAMErr__Sequence * input,
  eufs_msgs__msg__SLAMErr__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__SLAMErr);
    eufs_msgs__msg__SLAMErr * data =
      (eufs_msgs__msg__SLAMErr *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__SLAMErr__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__SLAMErr__fini(&data[i]);
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
    if (!eufs_msgs__msg__SLAMErr__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
