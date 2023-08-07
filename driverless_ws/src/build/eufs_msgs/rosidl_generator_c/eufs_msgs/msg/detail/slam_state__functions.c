// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/SLAMState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/slam_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `status`
#include "rosidl_runtime_c/string_functions.h"

bool
eufs_msgs__msg__SLAMState__init(eufs_msgs__msg__SLAMState * msg)
{
  if (!msg) {
    return false;
  }
  // loop_closed
  // laps
  // status
  if (!rosidl_runtime_c__String__init(&msg->status)) {
    eufs_msgs__msg__SLAMState__fini(msg);
    return false;
  }
  // state
  return true;
}

void
eufs_msgs__msg__SLAMState__fini(eufs_msgs__msg__SLAMState * msg)
{
  if (!msg) {
    return;
  }
  // loop_closed
  // laps
  // status
  rosidl_runtime_c__String__fini(&msg->status);
  // state
}

bool
eufs_msgs__msg__SLAMState__are_equal(const eufs_msgs__msg__SLAMState * lhs, const eufs_msgs__msg__SLAMState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // loop_closed
  if (lhs->loop_closed != rhs->loop_closed) {
    return false;
  }
  // laps
  if (lhs->laps != rhs->laps) {
    return false;
  }
  // status
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->status), &(rhs->status)))
  {
    return false;
  }
  // state
  if (lhs->state != rhs->state) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__SLAMState__copy(
  const eufs_msgs__msg__SLAMState * input,
  eufs_msgs__msg__SLAMState * output)
{
  if (!input || !output) {
    return false;
  }
  // loop_closed
  output->loop_closed = input->loop_closed;
  // laps
  output->laps = input->laps;
  // status
  if (!rosidl_runtime_c__String__copy(
      &(input->status), &(output->status)))
  {
    return false;
  }
  // state
  output->state = input->state;
  return true;
}

eufs_msgs__msg__SLAMState *
eufs_msgs__msg__SLAMState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__SLAMState * msg = (eufs_msgs__msg__SLAMState *)allocator.allocate(sizeof(eufs_msgs__msg__SLAMState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__SLAMState));
  bool success = eufs_msgs__msg__SLAMState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__SLAMState__destroy(eufs_msgs__msg__SLAMState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__SLAMState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__SLAMState__Sequence__init(eufs_msgs__msg__SLAMState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__SLAMState * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__SLAMState *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__SLAMState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__SLAMState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__SLAMState__fini(&data[i - 1]);
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
eufs_msgs__msg__SLAMState__Sequence__fini(eufs_msgs__msg__SLAMState__Sequence * array)
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
      eufs_msgs__msg__SLAMState__fini(&array->data[i]);
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

eufs_msgs__msg__SLAMState__Sequence *
eufs_msgs__msg__SLAMState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__SLAMState__Sequence * array = (eufs_msgs__msg__SLAMState__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__SLAMState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__SLAMState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__SLAMState__Sequence__destroy(eufs_msgs__msg__SLAMState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__SLAMState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__SLAMState__Sequence__are_equal(const eufs_msgs__msg__SLAMState__Sequence * lhs, const eufs_msgs__msg__SLAMState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__SLAMState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__SLAMState__Sequence__copy(
  const eufs_msgs__msg__SLAMState__Sequence * input,
  eufs_msgs__msg__SLAMState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__SLAMState);
    eufs_msgs__msg__SLAMState * data =
      (eufs_msgs__msg__SLAMState *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__SLAMState__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__SLAMState__fini(&data[i]);
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
    if (!eufs_msgs__msg__SLAMState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
