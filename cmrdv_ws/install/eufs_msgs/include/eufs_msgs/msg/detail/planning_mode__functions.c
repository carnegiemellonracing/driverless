// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/PlanningMode.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/planning_mode__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>


bool
eufs_msgs__msg__PlanningMode__init(eufs_msgs__msg__PlanningMode * msg)
{
  if (!msg) {
    return false;
  }
  // mode
  return true;
}

void
eufs_msgs__msg__PlanningMode__fini(eufs_msgs__msg__PlanningMode * msg)
{
  if (!msg) {
    return;
  }
  // mode
}

bool
eufs_msgs__msg__PlanningMode__are_equal(const eufs_msgs__msg__PlanningMode * lhs, const eufs_msgs__msg__PlanningMode * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // mode
  if (lhs->mode != rhs->mode) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__PlanningMode__copy(
  const eufs_msgs__msg__PlanningMode * input,
  eufs_msgs__msg__PlanningMode * output)
{
  if (!input || !output) {
    return false;
  }
  // mode
  output->mode = input->mode;
  return true;
}

eufs_msgs__msg__PlanningMode *
eufs_msgs__msg__PlanningMode__create()
{
  eufs_msgs__msg__PlanningMode * msg = (eufs_msgs__msg__PlanningMode *)malloc(sizeof(eufs_msgs__msg__PlanningMode));
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__PlanningMode));
  bool success = eufs_msgs__msg__PlanningMode__init(msg);
  if (!success) {
    free(msg);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__PlanningMode__destroy(eufs_msgs__msg__PlanningMode * msg)
{
  if (msg) {
    eufs_msgs__msg__PlanningMode__fini(msg);
  }
  free(msg);
}


bool
eufs_msgs__msg__PlanningMode__Sequence__init(eufs_msgs__msg__PlanningMode__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  eufs_msgs__msg__PlanningMode * data = NULL;
  if (size) {
    data = (eufs_msgs__msg__PlanningMode *)calloc(size, sizeof(eufs_msgs__msg__PlanningMode));
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__PlanningMode__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__PlanningMode__fini(&data[i - 1]);
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
eufs_msgs__msg__PlanningMode__Sequence__fini(eufs_msgs__msg__PlanningMode__Sequence * array)
{
  if (!array) {
    return;
  }
  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      eufs_msgs__msg__PlanningMode__fini(&array->data[i]);
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

eufs_msgs__msg__PlanningMode__Sequence *
eufs_msgs__msg__PlanningMode__Sequence__create(size_t size)
{
  eufs_msgs__msg__PlanningMode__Sequence * array = (eufs_msgs__msg__PlanningMode__Sequence *)malloc(sizeof(eufs_msgs__msg__PlanningMode__Sequence));
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__PlanningMode__Sequence__init(array, size);
  if (!success) {
    free(array);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__PlanningMode__Sequence__destroy(eufs_msgs__msg__PlanningMode__Sequence * array)
{
  if (array) {
    eufs_msgs__msg__PlanningMode__Sequence__fini(array);
  }
  free(array);
}

bool
eufs_msgs__msg__PlanningMode__Sequence__are_equal(const eufs_msgs__msg__PlanningMode__Sequence * lhs, const eufs_msgs__msg__PlanningMode__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__PlanningMode__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__PlanningMode__Sequence__copy(
  const eufs_msgs__msg__PlanningMode__Sequence * input,
  eufs_msgs__msg__PlanningMode__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__PlanningMode);
    eufs_msgs__msg__PlanningMode * data =
      (eufs_msgs__msg__PlanningMode *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__PlanningMode__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__PlanningMode__fini(&data[i]);
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
    if (!eufs_msgs__msg__PlanningMode__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
