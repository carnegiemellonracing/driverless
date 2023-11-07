// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cmrdv_interfaces:msg/Brakes.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/brakes__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `last_fired`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
cmrdv_interfaces__msg__Brakes__init(cmrdv_interfaces__msg__Brakes * msg)
{
  if (!msg) {
    return false;
  }
  // braking
  // last_fired
  if (!builtin_interfaces__msg__Time__init(&msg->last_fired)) {
    cmrdv_interfaces__msg__Brakes__fini(msg);
    return false;
  }
  return true;
}

void
cmrdv_interfaces__msg__Brakes__fini(cmrdv_interfaces__msg__Brakes * msg)
{
  if (!msg) {
    return;
  }
  // braking
  // last_fired
  builtin_interfaces__msg__Time__fini(&msg->last_fired);
}

bool
cmrdv_interfaces__msg__Brakes__are_equal(const cmrdv_interfaces__msg__Brakes * lhs, const cmrdv_interfaces__msg__Brakes * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // braking
  if (lhs->braking != rhs->braking) {
    return false;
  }
  // last_fired
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->last_fired), &(rhs->last_fired)))
  {
    return false;
  }
  return true;
}

bool
cmrdv_interfaces__msg__Brakes__copy(
  const cmrdv_interfaces__msg__Brakes * input,
  cmrdv_interfaces__msg__Brakes * output)
{
  if (!input || !output) {
    return false;
  }
  // braking
  output->braking = input->braking;
  // last_fired
  if (!builtin_interfaces__msg__Time__copy(
      &(input->last_fired), &(output->last_fired)))
  {
    return false;
  }
  return true;
}

cmrdv_interfaces__msg__Brakes *
cmrdv_interfaces__msg__Brakes__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__Brakes * msg = (cmrdv_interfaces__msg__Brakes *)allocator.allocate(sizeof(cmrdv_interfaces__msg__Brakes), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cmrdv_interfaces__msg__Brakes));
  bool success = cmrdv_interfaces__msg__Brakes__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cmrdv_interfaces__msg__Brakes__destroy(cmrdv_interfaces__msg__Brakes * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cmrdv_interfaces__msg__Brakes__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cmrdv_interfaces__msg__Brakes__Sequence__init(cmrdv_interfaces__msg__Brakes__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__Brakes * data = NULL;

  if (size) {
    data = (cmrdv_interfaces__msg__Brakes *)allocator.zero_allocate(size, sizeof(cmrdv_interfaces__msg__Brakes), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cmrdv_interfaces__msg__Brakes__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cmrdv_interfaces__msg__Brakes__fini(&data[i - 1]);
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
cmrdv_interfaces__msg__Brakes__Sequence__fini(cmrdv_interfaces__msg__Brakes__Sequence * array)
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
      cmrdv_interfaces__msg__Brakes__fini(&array->data[i]);
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

cmrdv_interfaces__msg__Brakes__Sequence *
cmrdv_interfaces__msg__Brakes__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__Brakes__Sequence * array = (cmrdv_interfaces__msg__Brakes__Sequence *)allocator.allocate(sizeof(cmrdv_interfaces__msg__Brakes__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cmrdv_interfaces__msg__Brakes__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cmrdv_interfaces__msg__Brakes__Sequence__destroy(cmrdv_interfaces__msg__Brakes__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cmrdv_interfaces__msg__Brakes__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cmrdv_interfaces__msg__Brakes__Sequence__are_equal(const cmrdv_interfaces__msg__Brakes__Sequence * lhs, const cmrdv_interfaces__msg__Brakes__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cmrdv_interfaces__msg__Brakes__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cmrdv_interfaces__msg__Brakes__Sequence__copy(
  const cmrdv_interfaces__msg__Brakes__Sequence * input,
  cmrdv_interfaces__msg__Brakes__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cmrdv_interfaces__msg__Brakes);
    cmrdv_interfaces__msg__Brakes * data =
      (cmrdv_interfaces__msg__Brakes *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cmrdv_interfaces__msg__Brakes__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          cmrdv_interfaces__msg__Brakes__fini(&data[i]);
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
    if (!cmrdv_interfaces__msg__Brakes__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
