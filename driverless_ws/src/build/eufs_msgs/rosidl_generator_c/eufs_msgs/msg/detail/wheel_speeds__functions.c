// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/WheelSpeeds.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/wheel_speeds__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
eufs_msgs__msg__WheelSpeeds__init(eufs_msgs__msg__WheelSpeeds * msg)
{
  if (!msg) {
    return false;
  }
  // steering
  // lf_speed
  // rf_speed
  // lb_speed
  // rb_speed
  return true;
}

void
eufs_msgs__msg__WheelSpeeds__fini(eufs_msgs__msg__WheelSpeeds * msg)
{
  if (!msg) {
    return;
  }
  // steering
  // lf_speed
  // rf_speed
  // lb_speed
  // rb_speed
}

bool
eufs_msgs__msg__WheelSpeeds__are_equal(const eufs_msgs__msg__WheelSpeeds * lhs, const eufs_msgs__msg__WheelSpeeds * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // steering
  if (lhs->steering != rhs->steering) {
    return false;
  }
  // lf_speed
  if (lhs->lf_speed != rhs->lf_speed) {
    return false;
  }
  // rf_speed
  if (lhs->rf_speed != rhs->rf_speed) {
    return false;
  }
  // lb_speed
  if (lhs->lb_speed != rhs->lb_speed) {
    return false;
  }
  // rb_speed
  if (lhs->rb_speed != rhs->rb_speed) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__WheelSpeeds__copy(
  const eufs_msgs__msg__WheelSpeeds * input,
  eufs_msgs__msg__WheelSpeeds * output)
{
  if (!input || !output) {
    return false;
  }
  // steering
  output->steering = input->steering;
  // lf_speed
  output->lf_speed = input->lf_speed;
  // rf_speed
  output->rf_speed = input->rf_speed;
  // lb_speed
  output->lb_speed = input->lb_speed;
  // rb_speed
  output->rb_speed = input->rb_speed;
  return true;
}

eufs_msgs__msg__WheelSpeeds *
eufs_msgs__msg__WheelSpeeds__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__WheelSpeeds * msg = (eufs_msgs__msg__WheelSpeeds *)allocator.allocate(sizeof(eufs_msgs__msg__WheelSpeeds), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__WheelSpeeds));
  bool success = eufs_msgs__msg__WheelSpeeds__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__WheelSpeeds__destroy(eufs_msgs__msg__WheelSpeeds * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__WheelSpeeds__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__WheelSpeeds__Sequence__init(eufs_msgs__msg__WheelSpeeds__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__WheelSpeeds * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__WheelSpeeds *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__WheelSpeeds), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__WheelSpeeds__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__WheelSpeeds__fini(&data[i - 1]);
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
eufs_msgs__msg__WheelSpeeds__Sequence__fini(eufs_msgs__msg__WheelSpeeds__Sequence * array)
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
      eufs_msgs__msg__WheelSpeeds__fini(&array->data[i]);
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

eufs_msgs__msg__WheelSpeeds__Sequence *
eufs_msgs__msg__WheelSpeeds__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__WheelSpeeds__Sequence * array = (eufs_msgs__msg__WheelSpeeds__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__WheelSpeeds__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__WheelSpeeds__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__WheelSpeeds__Sequence__destroy(eufs_msgs__msg__WheelSpeeds__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__WheelSpeeds__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__WheelSpeeds__Sequence__are_equal(const eufs_msgs__msg__WheelSpeeds__Sequence * lhs, const eufs_msgs__msg__WheelSpeeds__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__WheelSpeeds__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__WheelSpeeds__Sequence__copy(
  const eufs_msgs__msg__WheelSpeeds__Sequence * input,
  eufs_msgs__msg__WheelSpeeds__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__WheelSpeeds);
    eufs_msgs__msg__WheelSpeeds * data =
      (eufs_msgs__msg__WheelSpeeds *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__WheelSpeeds__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__WheelSpeeds__fini(&data[i]);
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
    if (!eufs_msgs__msg__WheelSpeeds__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
