// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/Heartbeat.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/heartbeat__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>


bool
eufs_msgs__msg__Heartbeat__init(eufs_msgs__msg__Heartbeat * msg)
{
  if (!msg) {
    return false;
  }
  // id
  // data
  return true;
}

void
eufs_msgs__msg__Heartbeat__fini(eufs_msgs__msg__Heartbeat * msg)
{
  if (!msg) {
    return;
  }
  // id
  // data
}

bool
eufs_msgs__msg__Heartbeat__are_equal(const eufs_msgs__msg__Heartbeat * lhs, const eufs_msgs__msg__Heartbeat * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  // data
  if (lhs->data != rhs->data) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__Heartbeat__copy(
  const eufs_msgs__msg__Heartbeat * input,
  eufs_msgs__msg__Heartbeat * output)
{
  if (!input || !output) {
    return false;
  }
  // id
  output->id = input->id;
  // data
  output->data = input->data;
  return true;
}

eufs_msgs__msg__Heartbeat *
eufs_msgs__msg__Heartbeat__create()
{
  eufs_msgs__msg__Heartbeat * msg = (eufs_msgs__msg__Heartbeat *)malloc(sizeof(eufs_msgs__msg__Heartbeat));
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__Heartbeat));
  bool success = eufs_msgs__msg__Heartbeat__init(msg);
  if (!success) {
    free(msg);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__Heartbeat__destroy(eufs_msgs__msg__Heartbeat * msg)
{
  if (msg) {
    eufs_msgs__msg__Heartbeat__fini(msg);
  }
  free(msg);
}


bool
eufs_msgs__msg__Heartbeat__Sequence__init(eufs_msgs__msg__Heartbeat__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  eufs_msgs__msg__Heartbeat * data = NULL;
  if (size) {
    data = (eufs_msgs__msg__Heartbeat *)calloc(size, sizeof(eufs_msgs__msg__Heartbeat));
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__Heartbeat__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__Heartbeat__fini(&data[i - 1]);
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
eufs_msgs__msg__Heartbeat__Sequence__fini(eufs_msgs__msg__Heartbeat__Sequence * array)
{
  if (!array) {
    return;
  }
  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      eufs_msgs__msg__Heartbeat__fini(&array->data[i]);
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

eufs_msgs__msg__Heartbeat__Sequence *
eufs_msgs__msg__Heartbeat__Sequence__create(size_t size)
{
  eufs_msgs__msg__Heartbeat__Sequence * array = (eufs_msgs__msg__Heartbeat__Sequence *)malloc(sizeof(eufs_msgs__msg__Heartbeat__Sequence));
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__Heartbeat__Sequence__init(array, size);
  if (!success) {
    free(array);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__Heartbeat__Sequence__destroy(eufs_msgs__msg__Heartbeat__Sequence * array)
{
  if (array) {
    eufs_msgs__msg__Heartbeat__Sequence__fini(array);
  }
  free(array);
}

bool
eufs_msgs__msg__Heartbeat__Sequence__are_equal(const eufs_msgs__msg__Heartbeat__Sequence * lhs, const eufs_msgs__msg__Heartbeat__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__Heartbeat__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__Heartbeat__Sequence__copy(
  const eufs_msgs__msg__Heartbeat__Sequence * input,
  eufs_msgs__msg__Heartbeat__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__Heartbeat);
    eufs_msgs__msg__Heartbeat * data =
      (eufs_msgs__msg__Heartbeat *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__Heartbeat__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__Heartbeat__fini(&data[i]);
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
    if (!eufs_msgs__msg__Heartbeat__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
