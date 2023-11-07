// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cmrdv_interfaces:msg/ConePositions.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/cone_positions__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `cone_positions`
#include "std_msgs/msg/detail/float32__functions.h"

bool
cmrdv_interfaces__msg__ConePositions__init(cmrdv_interfaces__msg__ConePositions * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    cmrdv_interfaces__msg__ConePositions__fini(msg);
    return false;
  }
  // cone_positions
  if (!std_msgs__msg__Float32__Sequence__init(&msg->cone_positions, 0)) {
    cmrdv_interfaces__msg__ConePositions__fini(msg);
    return false;
  }
  return true;
}

void
cmrdv_interfaces__msg__ConePositions__fini(cmrdv_interfaces__msg__ConePositions * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // cone_positions
  std_msgs__msg__Float32__Sequence__fini(&msg->cone_positions);
}

bool
cmrdv_interfaces__msg__ConePositions__are_equal(const cmrdv_interfaces__msg__ConePositions * lhs, const cmrdv_interfaces__msg__ConePositions * rhs)
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
  // cone_positions
  if (!std_msgs__msg__Float32__Sequence__are_equal(
      &(lhs->cone_positions), &(rhs->cone_positions)))
  {
    return false;
  }
  return true;
}

bool
cmrdv_interfaces__msg__ConePositions__copy(
  const cmrdv_interfaces__msg__ConePositions * input,
  cmrdv_interfaces__msg__ConePositions * output)
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
  // cone_positions
  if (!std_msgs__msg__Float32__Sequence__copy(
      &(input->cone_positions), &(output->cone_positions)))
  {
    return false;
  }
  return true;
}

cmrdv_interfaces__msg__ConePositions *
cmrdv_interfaces__msg__ConePositions__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__ConePositions * msg = (cmrdv_interfaces__msg__ConePositions *)allocator.allocate(sizeof(cmrdv_interfaces__msg__ConePositions), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cmrdv_interfaces__msg__ConePositions));
  bool success = cmrdv_interfaces__msg__ConePositions__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cmrdv_interfaces__msg__ConePositions__destroy(cmrdv_interfaces__msg__ConePositions * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cmrdv_interfaces__msg__ConePositions__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cmrdv_interfaces__msg__ConePositions__Sequence__init(cmrdv_interfaces__msg__ConePositions__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__ConePositions * data = NULL;

  if (size) {
    data = (cmrdv_interfaces__msg__ConePositions *)allocator.zero_allocate(size, sizeof(cmrdv_interfaces__msg__ConePositions), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cmrdv_interfaces__msg__ConePositions__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cmrdv_interfaces__msg__ConePositions__fini(&data[i - 1]);
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
cmrdv_interfaces__msg__ConePositions__Sequence__fini(cmrdv_interfaces__msg__ConePositions__Sequence * array)
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
      cmrdv_interfaces__msg__ConePositions__fini(&array->data[i]);
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

cmrdv_interfaces__msg__ConePositions__Sequence *
cmrdv_interfaces__msg__ConePositions__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__ConePositions__Sequence * array = (cmrdv_interfaces__msg__ConePositions__Sequence *)allocator.allocate(sizeof(cmrdv_interfaces__msg__ConePositions__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cmrdv_interfaces__msg__ConePositions__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cmrdv_interfaces__msg__ConePositions__Sequence__destroy(cmrdv_interfaces__msg__ConePositions__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cmrdv_interfaces__msg__ConePositions__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cmrdv_interfaces__msg__ConePositions__Sequence__are_equal(const cmrdv_interfaces__msg__ConePositions__Sequence * lhs, const cmrdv_interfaces__msg__ConePositions__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cmrdv_interfaces__msg__ConePositions__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cmrdv_interfaces__msg__ConePositions__Sequence__copy(
  const cmrdv_interfaces__msg__ConePositions__Sequence * input,
  cmrdv_interfaces__msg__ConePositions__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cmrdv_interfaces__msg__ConePositions);
    cmrdv_interfaces__msg__ConePositions * data =
      (cmrdv_interfaces__msg__ConePositions *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cmrdv_interfaces__msg__ConePositions__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          cmrdv_interfaces__msg__ConePositions__fini(&data[i]);
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
    if (!cmrdv_interfaces__msg__ConePositions__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
