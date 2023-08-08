// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/FullState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/full_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
eufs_msgs__msg__FullState__init(eufs_msgs__msg__FullState * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__FullState__fini(msg);
    return false;
  }
  // x_pos
  // y_pos
  // yaw
  // roll
  // u_x
  // u_y
  // yaw_mder
  // front_throttle
  // rear_throttle
  // steering
  return true;
}

void
eufs_msgs__msg__FullState__fini(eufs_msgs__msg__FullState * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // x_pos
  // y_pos
  // yaw
  // roll
  // u_x
  // u_y
  // yaw_mder
  // front_throttle
  // rear_throttle
  // steering
}

bool
eufs_msgs__msg__FullState__are_equal(const eufs_msgs__msg__FullState * lhs, const eufs_msgs__msg__FullState * rhs)
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
  // x_pos
  if (lhs->x_pos != rhs->x_pos) {
    return false;
  }
  // y_pos
  if (lhs->y_pos != rhs->y_pos) {
    return false;
  }
  // yaw
  if (lhs->yaw != rhs->yaw) {
    return false;
  }
  // roll
  if (lhs->roll != rhs->roll) {
    return false;
  }
  // u_x
  if (lhs->u_x != rhs->u_x) {
    return false;
  }
  // u_y
  if (lhs->u_y != rhs->u_y) {
    return false;
  }
  // yaw_mder
  if (lhs->yaw_mder != rhs->yaw_mder) {
    return false;
  }
  // front_throttle
  if (lhs->front_throttle != rhs->front_throttle) {
    return false;
  }
  // rear_throttle
  if (lhs->rear_throttle != rhs->rear_throttle) {
    return false;
  }
  // steering
  if (lhs->steering != rhs->steering) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__FullState__copy(
  const eufs_msgs__msg__FullState * input,
  eufs_msgs__msg__FullState * output)
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
  // x_pos
  output->x_pos = input->x_pos;
  // y_pos
  output->y_pos = input->y_pos;
  // yaw
  output->yaw = input->yaw;
  // roll
  output->roll = input->roll;
  // u_x
  output->u_x = input->u_x;
  // u_y
  output->u_y = input->u_y;
  // yaw_mder
  output->yaw_mder = input->yaw_mder;
  // front_throttle
  output->front_throttle = input->front_throttle;
  // rear_throttle
  output->rear_throttle = input->rear_throttle;
  // steering
  output->steering = input->steering;
  return true;
}

eufs_msgs__msg__FullState *
eufs_msgs__msg__FullState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__FullState * msg = (eufs_msgs__msg__FullState *)allocator.allocate(sizeof(eufs_msgs__msg__FullState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__FullState));
  bool success = eufs_msgs__msg__FullState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__FullState__destroy(eufs_msgs__msg__FullState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__FullState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__FullState__Sequence__init(eufs_msgs__msg__FullState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__FullState * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__FullState *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__FullState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__FullState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__FullState__fini(&data[i - 1]);
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
eufs_msgs__msg__FullState__Sequence__fini(eufs_msgs__msg__FullState__Sequence * array)
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
      eufs_msgs__msg__FullState__fini(&array->data[i]);
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

eufs_msgs__msg__FullState__Sequence *
eufs_msgs__msg__FullState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__FullState__Sequence * array = (eufs_msgs__msg__FullState__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__FullState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__FullState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__FullState__Sequence__destroy(eufs_msgs__msg__FullState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__FullState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__FullState__Sequence__are_equal(const eufs_msgs__msg__FullState__Sequence * lhs, const eufs_msgs__msg__FullState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__FullState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__FullState__Sequence__copy(
  const eufs_msgs__msg__FullState__Sequence * input,
  eufs_msgs__msg__FullState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__FullState);
    eufs_msgs__msg__FullState * data =
      (eufs_msgs__msg__FullState *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__FullState__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__FullState__fini(&data[i]);
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
    if (!eufs_msgs__msg__FullState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
