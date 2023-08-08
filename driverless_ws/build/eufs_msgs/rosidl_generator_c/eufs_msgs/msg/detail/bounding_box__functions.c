// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/BoundingBox.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/bounding_box__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `color`
#include "rosidl_runtime_c/string_functions.h"

bool
eufs_msgs__msg__BoundingBox__init(eufs_msgs__msg__BoundingBox * msg)
{
  if (!msg) {
    return false;
  }
  // color
  if (!rosidl_runtime_c__String__init(&msg->color)) {
    eufs_msgs__msg__BoundingBox__fini(msg);
    return false;
  }
  // probability
  // type
  // xmin
  // ymin
  // xmax
  // ymax
  return true;
}

void
eufs_msgs__msg__BoundingBox__fini(eufs_msgs__msg__BoundingBox * msg)
{
  if (!msg) {
    return;
  }
  // color
  rosidl_runtime_c__String__fini(&msg->color);
  // probability
  // type
  // xmin
  // ymin
  // xmax
  // ymax
}

bool
eufs_msgs__msg__BoundingBox__are_equal(const eufs_msgs__msg__BoundingBox * lhs, const eufs_msgs__msg__BoundingBox * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // color
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->color), &(rhs->color)))
  {
    return false;
  }
  // probability
  if (lhs->probability != rhs->probability) {
    return false;
  }
  // type
  if (lhs->type != rhs->type) {
    return false;
  }
  // xmin
  if (lhs->xmin != rhs->xmin) {
    return false;
  }
  // ymin
  if (lhs->ymin != rhs->ymin) {
    return false;
  }
  // xmax
  if (lhs->xmax != rhs->xmax) {
    return false;
  }
  // ymax
  if (lhs->ymax != rhs->ymax) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__BoundingBox__copy(
  const eufs_msgs__msg__BoundingBox * input,
  eufs_msgs__msg__BoundingBox * output)
{
  if (!input || !output) {
    return false;
  }
  // color
  if (!rosidl_runtime_c__String__copy(
      &(input->color), &(output->color)))
  {
    return false;
  }
  // probability
  output->probability = input->probability;
  // type
  output->type = input->type;
  // xmin
  output->xmin = input->xmin;
  // ymin
  output->ymin = input->ymin;
  // xmax
  output->xmax = input->xmax;
  // ymax
  output->ymax = input->ymax;
  return true;
}

eufs_msgs__msg__BoundingBox *
eufs_msgs__msg__BoundingBox__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__BoundingBox * msg = (eufs_msgs__msg__BoundingBox *)allocator.allocate(sizeof(eufs_msgs__msg__BoundingBox), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__BoundingBox));
  bool success = eufs_msgs__msg__BoundingBox__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__BoundingBox__destroy(eufs_msgs__msg__BoundingBox * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__BoundingBox__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__BoundingBox__Sequence__init(eufs_msgs__msg__BoundingBox__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__BoundingBox * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__BoundingBox *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__BoundingBox), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__BoundingBox__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__BoundingBox__fini(&data[i - 1]);
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
eufs_msgs__msg__BoundingBox__Sequence__fini(eufs_msgs__msg__BoundingBox__Sequence * array)
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
      eufs_msgs__msg__BoundingBox__fini(&array->data[i]);
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

eufs_msgs__msg__BoundingBox__Sequence *
eufs_msgs__msg__BoundingBox__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__BoundingBox__Sequence * array = (eufs_msgs__msg__BoundingBox__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__BoundingBox__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__BoundingBox__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__BoundingBox__Sequence__destroy(eufs_msgs__msg__BoundingBox__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__BoundingBox__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__BoundingBox__Sequence__are_equal(const eufs_msgs__msg__BoundingBox__Sequence * lhs, const eufs_msgs__msg__BoundingBox__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__BoundingBox__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__BoundingBox__Sequence__copy(
  const eufs_msgs__msg__BoundingBox__Sequence * input,
  eufs_msgs__msg__BoundingBox__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__BoundingBox);
    eufs_msgs__msg__BoundingBox * data =
      (eufs_msgs__msg__BoundingBox *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__BoundingBox__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__BoundingBox__fini(&data[i]);
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
    if (!eufs_msgs__msg__BoundingBox__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
