// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cmrdv_interfaces:msg/ControlAction.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/control_action__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
cmrdv_interfaces__msg__ControlAction__init(cmrdv_interfaces__msg__ControlAction * msg)
{
  if (!msg) {
    return false;
  }
  // wheel_speed
  // swangle
  return true;
}

void
cmrdv_interfaces__msg__ControlAction__fini(cmrdv_interfaces__msg__ControlAction * msg)
{
  if (!msg) {
    return;
  }
  // wheel_speed
  // swangle
}

bool
cmrdv_interfaces__msg__ControlAction__are_equal(const cmrdv_interfaces__msg__ControlAction * lhs, const cmrdv_interfaces__msg__ControlAction * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // wheel_speed
  if (lhs->wheel_speed != rhs->wheel_speed) {
    return false;
  }
  // swangle
  if (lhs->swangle != rhs->swangle) {
    return false;
  }
  return true;
}

bool
cmrdv_interfaces__msg__ControlAction__copy(
  const cmrdv_interfaces__msg__ControlAction * input,
  cmrdv_interfaces__msg__ControlAction * output)
{
  if (!input || !output) {
    return false;
  }
  // wheel_speed
  output->wheel_speed = input->wheel_speed;
  // swangle
  output->swangle = input->swangle;
  return true;
}

cmrdv_interfaces__msg__ControlAction *
cmrdv_interfaces__msg__ControlAction__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__ControlAction * msg = (cmrdv_interfaces__msg__ControlAction *)allocator.allocate(sizeof(cmrdv_interfaces__msg__ControlAction), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cmrdv_interfaces__msg__ControlAction));
  bool success = cmrdv_interfaces__msg__ControlAction__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cmrdv_interfaces__msg__ControlAction__destroy(cmrdv_interfaces__msg__ControlAction * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cmrdv_interfaces__msg__ControlAction__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cmrdv_interfaces__msg__ControlAction__Sequence__init(cmrdv_interfaces__msg__ControlAction__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__ControlAction * data = NULL;

  if (size) {
    data = (cmrdv_interfaces__msg__ControlAction *)allocator.zero_allocate(size, sizeof(cmrdv_interfaces__msg__ControlAction), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cmrdv_interfaces__msg__ControlAction__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cmrdv_interfaces__msg__ControlAction__fini(&data[i - 1]);
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
cmrdv_interfaces__msg__ControlAction__Sequence__fini(cmrdv_interfaces__msg__ControlAction__Sequence * array)
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
      cmrdv_interfaces__msg__ControlAction__fini(&array->data[i]);
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

cmrdv_interfaces__msg__ControlAction__Sequence *
cmrdv_interfaces__msg__ControlAction__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__ControlAction__Sequence * array = (cmrdv_interfaces__msg__ControlAction__Sequence *)allocator.allocate(sizeof(cmrdv_interfaces__msg__ControlAction__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cmrdv_interfaces__msg__ControlAction__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cmrdv_interfaces__msg__ControlAction__Sequence__destroy(cmrdv_interfaces__msg__ControlAction__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cmrdv_interfaces__msg__ControlAction__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cmrdv_interfaces__msg__ControlAction__Sequence__are_equal(const cmrdv_interfaces__msg__ControlAction__Sequence * lhs, const cmrdv_interfaces__msg__ControlAction__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cmrdv_interfaces__msg__ControlAction__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cmrdv_interfaces__msg__ControlAction__Sequence__copy(
  const cmrdv_interfaces__msg__ControlAction__Sequence * input,
  cmrdv_interfaces__msg__ControlAction__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cmrdv_interfaces__msg__ControlAction);
    cmrdv_interfaces__msg__ControlAction * data =
      (cmrdv_interfaces__msg__ControlAction *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cmrdv_interfaces__msg__ControlAction__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          cmrdv_interfaces__msg__ControlAction__fini(&data[i]);
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
    if (!cmrdv_interfaces__msg__ControlAction__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
