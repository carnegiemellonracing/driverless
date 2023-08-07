// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/ConeArray.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/cone_array__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `blue_cones`
// Member `yellow_cones`
// Member `orange_cones`
// Member `big_orange_cones`
// Member `unknown_color_cones`
#include "geometry_msgs/msg/detail/point__functions.h"

bool
eufs_msgs__msg__ConeArray__init(eufs_msgs__msg__ConeArray * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__ConeArray__fini(msg);
    return false;
  }
  // blue_cones
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->blue_cones, 0)) {
    eufs_msgs__msg__ConeArray__fini(msg);
    return false;
  }
  // yellow_cones
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->yellow_cones, 0)) {
    eufs_msgs__msg__ConeArray__fini(msg);
    return false;
  }
  // orange_cones
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->orange_cones, 0)) {
    eufs_msgs__msg__ConeArray__fini(msg);
    return false;
  }
  // big_orange_cones
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->big_orange_cones, 0)) {
    eufs_msgs__msg__ConeArray__fini(msg);
    return false;
  }
  // unknown_color_cones
  if (!geometry_msgs__msg__Point__Sequence__init(&msg->unknown_color_cones, 0)) {
    eufs_msgs__msg__ConeArray__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__msg__ConeArray__fini(eufs_msgs__msg__ConeArray * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // blue_cones
  geometry_msgs__msg__Point__Sequence__fini(&msg->blue_cones);
  // yellow_cones
  geometry_msgs__msg__Point__Sequence__fini(&msg->yellow_cones);
  // orange_cones
  geometry_msgs__msg__Point__Sequence__fini(&msg->orange_cones);
  // big_orange_cones
  geometry_msgs__msg__Point__Sequence__fini(&msg->big_orange_cones);
  // unknown_color_cones
  geometry_msgs__msg__Point__Sequence__fini(&msg->unknown_color_cones);
}

bool
eufs_msgs__msg__ConeArray__are_equal(const eufs_msgs__msg__ConeArray * lhs, const eufs_msgs__msg__ConeArray * rhs)
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
  // big_orange_cones
  if (!geometry_msgs__msg__Point__Sequence__are_equal(
      &(lhs->big_orange_cones), &(rhs->big_orange_cones)))
  {
    return false;
  }
  // unknown_color_cones
  if (!geometry_msgs__msg__Point__Sequence__are_equal(
      &(lhs->unknown_color_cones), &(rhs->unknown_color_cones)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__ConeArray__copy(
  const eufs_msgs__msg__ConeArray * input,
  eufs_msgs__msg__ConeArray * output)
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
  // big_orange_cones
  if (!geometry_msgs__msg__Point__Sequence__copy(
      &(input->big_orange_cones), &(output->big_orange_cones)))
  {
    return false;
  }
  // unknown_color_cones
  if (!geometry_msgs__msg__Point__Sequence__copy(
      &(input->unknown_color_cones), &(output->unknown_color_cones)))
  {
    return false;
  }
  return true;
}

eufs_msgs__msg__ConeArray *
eufs_msgs__msg__ConeArray__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__ConeArray * msg = (eufs_msgs__msg__ConeArray *)allocator.allocate(sizeof(eufs_msgs__msg__ConeArray), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__ConeArray));
  bool success = eufs_msgs__msg__ConeArray__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__ConeArray__destroy(eufs_msgs__msg__ConeArray * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__ConeArray__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__ConeArray__Sequence__init(eufs_msgs__msg__ConeArray__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__ConeArray * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__ConeArray *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__ConeArray), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__ConeArray__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__ConeArray__fini(&data[i - 1]);
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
eufs_msgs__msg__ConeArray__Sequence__fini(eufs_msgs__msg__ConeArray__Sequence * array)
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
      eufs_msgs__msg__ConeArray__fini(&array->data[i]);
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

eufs_msgs__msg__ConeArray__Sequence *
eufs_msgs__msg__ConeArray__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__ConeArray__Sequence * array = (eufs_msgs__msg__ConeArray__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__ConeArray__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__ConeArray__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__ConeArray__Sequence__destroy(eufs_msgs__msg__ConeArray__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__ConeArray__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__ConeArray__Sequence__are_equal(const eufs_msgs__msg__ConeArray__Sequence * lhs, const eufs_msgs__msg__ConeArray__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__ConeArray__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__ConeArray__Sequence__copy(
  const eufs_msgs__msg__ConeArray__Sequence * input,
  eufs_msgs__msg__ConeArray__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__ConeArray);
    eufs_msgs__msg__ConeArray * data =
      (eufs_msgs__msg__ConeArray *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__ConeArray__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__ConeArray__fini(&data[i]);
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
    if (!eufs_msgs__msg__ConeArray__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
