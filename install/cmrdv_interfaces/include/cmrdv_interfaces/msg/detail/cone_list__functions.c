// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cmrdv_interfaces:msg/ConeList.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/cone_list__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `blue_cones`
// Member `yellow_cones`
// Member `orange_cones`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
cmrdv_interfaces__msg__ConeList__init(cmrdv_interfaces__msg__ConeList * msg)
{
  if (!msg) {
    return false;
  }
  // blue_cones
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->blue_cones, 0)) {
    cmrdv_interfaces__msg__ConeList__fini(msg);
    return false;
  }
  // yellow_cones
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->yellow_cones, 0)) {
    cmrdv_interfaces__msg__ConeList__fini(msg);
    return false;
  }
  // orange_cones
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->orange_cones, 0)) {
    cmrdv_interfaces__msg__ConeList__fini(msg);
    return false;
  }
  return true;
}

void
cmrdv_interfaces__msg__ConeList__fini(cmrdv_interfaces__msg__ConeList * msg)
{
  if (!msg) {
    return;
  }
  // blue_cones
  geometry_msgs__msg__Point__Sequence__fini(&msg->blue_cones);
  // yellow_cones
  geometry_msgs__msg__Point__Sequence__fini(&msg->yellow_cones);
  // orange_cones
  geometry_msgs__msg__Point__Sequence__fini(&msg->orange_cones);
}

bool
cmrdv_interfaces__msg__ConeList__are_equal(const cmrdv_interfaces__msg__ConeList * lhs, const cmrdv_interfaces__msg__ConeList * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // blue_cones
  if (!geometry_msgs__msg__Point__Sequence__are_equal(
      &(lhs->blue_cones), &(rhs->blue_cones)))
  {
    return false;
  }
  // yellow_cones
  if (!geometry_msgs__msg__Point__Sequence__are_equal(
      &(lhs->yellow_cones), &(rhs->yellow_cones)))
  {
    return false;
  }
  // orange_cones
  if (!geometry_msgs__msg__Point__Sequence__are_equal(
      &(lhs->orange_cones), &(rhs->orange_cones)))
  {
    return false;
  }
  return true;
}

bool
cmrdv_interfaces__msg__ConeList__copy(
  const cmrdv_interfaces__msg__ConeList * input,
  cmrdv_interfaces__msg__ConeList * output)
{
  if (!input || !output) {
    return false;
  }
  // blue_cones
  if (!geometry_msgs__msg__Point__Sequence__copy(
      &(input->blue_cones), &(output->blue_cones)))
  {
    return false;
  }
  // yellow_cones
  if (!geometry_msgs__msg__Point__Sequence__copy(
      &(input->yellow_cones), &(output->yellow_cones)))
  {
    return false;
  }
  // orange_cones
  if (!geometry_msgs__msg__Point__Sequence__copy(
      &(input->orange_cones), &(output->orange_cones)))
  {
    return false;
  }
  return true;
}

cmrdv_interfaces__msg__ConeList *
cmrdv_interfaces__msg__ConeList__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__ConeList * msg = (cmrdv_interfaces__msg__ConeList *)allocator.allocate(sizeof(cmrdv_interfaces__msg__ConeList), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cmrdv_interfaces__msg__ConeList));
  bool success = cmrdv_interfaces__msg__ConeList__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cmrdv_interfaces__msg__ConeList__destroy(cmrdv_interfaces__msg__ConeList * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cmrdv_interfaces__msg__ConeList__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cmrdv_interfaces__msg__ConeList__Sequence__init(cmrdv_interfaces__msg__ConeList__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__ConeList * data = NULL;

  if (size) {
    data = (cmrdv_interfaces__msg__ConeList *)allocator.zero_allocate(size, sizeof(cmrdv_interfaces__msg__ConeList), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cmrdv_interfaces__msg__ConeList__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cmrdv_interfaces__msg__ConeList__fini(&data[i - 1]);
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
cmrdv_interfaces__msg__ConeList__Sequence__fini(cmrdv_interfaces__msg__ConeList__Sequence * array)
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
      cmrdv_interfaces__msg__ConeList__fini(&array->data[i]);
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

cmrdv_interfaces__msg__ConeList__Sequence *
cmrdv_interfaces__msg__ConeList__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__ConeList__Sequence * array = (cmrdv_interfaces__msg__ConeList__Sequence *)allocator.allocate(sizeof(cmrdv_interfaces__msg__ConeList__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cmrdv_interfaces__msg__ConeList__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cmrdv_interfaces__msg__ConeList__Sequence__destroy(cmrdv_interfaces__msg__ConeList__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cmrdv_interfaces__msg__ConeList__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cmrdv_interfaces__msg__ConeList__Sequence__are_equal(const cmrdv_interfaces__msg__ConeList__Sequence * lhs, const cmrdv_interfaces__msg__ConeList__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cmrdv_interfaces__msg__ConeList__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cmrdv_interfaces__msg__ConeList__Sequence__copy(
  const cmrdv_interfaces__msg__ConeList__Sequence * input,
  cmrdv_interfaces__msg__ConeList__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cmrdv_interfaces__msg__ConeList);
    cmrdv_interfaces__msg__ConeList * data =
      (cmrdv_interfaces__msg__ConeList *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cmrdv_interfaces__msg__ConeList__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          cmrdv_interfaces__msg__ConeList__fini(&data[i]);
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
    if (!cmrdv_interfaces__msg__ConeList__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
