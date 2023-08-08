// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/SystemStatus.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/system_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `topic_statuses`
#include "eufs_msgs/msg/detail/topic_status__functions.h"

bool
eufs_msgs__msg__SystemStatus__init(eufs_msgs__msg__SystemStatus * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__SystemStatus__fini(msg);
    return false;
  }
  // topic_statuses
  if (!eufs_msgs__msg__TopicStatus__Sequence__init(&msg->topic_statuses, 0)) {
    eufs_msgs__msg__SystemStatus__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__msg__SystemStatus__fini(eufs_msgs__msg__SystemStatus * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // topic_statuses
  eufs_msgs__msg__TopicStatus__Sequence__fini(&msg->topic_statuses);
}

bool
eufs_msgs__msg__SystemStatus__are_equal(const eufs_msgs__msg__SystemStatus * lhs, const eufs_msgs__msg__SystemStatus * rhs)
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
  // topic_statuses
  if (!eufs_msgs__msg__TopicStatus__Sequence__are_equal(
      &(lhs->topic_statuses), &(rhs->topic_statuses)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__SystemStatus__copy(
  const eufs_msgs__msg__SystemStatus * input,
  eufs_msgs__msg__SystemStatus * output)
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
  // topic_statuses
  if (!eufs_msgs__msg__TopicStatus__Sequence__copy(
      &(input->topic_statuses), &(output->topic_statuses)))
  {
    return false;
  }
  return true;
}

eufs_msgs__msg__SystemStatus *
eufs_msgs__msg__SystemStatus__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__SystemStatus * msg = (eufs_msgs__msg__SystemStatus *)allocator.allocate(sizeof(eufs_msgs__msg__SystemStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__SystemStatus));
  bool success = eufs_msgs__msg__SystemStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__SystemStatus__destroy(eufs_msgs__msg__SystemStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__SystemStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__SystemStatus__Sequence__init(eufs_msgs__msg__SystemStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__SystemStatus * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__SystemStatus *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__SystemStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__SystemStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__SystemStatus__fini(&data[i - 1]);
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
eufs_msgs__msg__SystemStatus__Sequence__fini(eufs_msgs__msg__SystemStatus__Sequence * array)
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
      eufs_msgs__msg__SystemStatus__fini(&array->data[i]);
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

eufs_msgs__msg__SystemStatus__Sequence *
eufs_msgs__msg__SystemStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__SystemStatus__Sequence * array = (eufs_msgs__msg__SystemStatus__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__SystemStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__SystemStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__SystemStatus__Sequence__destroy(eufs_msgs__msg__SystemStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__SystemStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__SystemStatus__Sequence__are_equal(const eufs_msgs__msg__SystemStatus__Sequence * lhs, const eufs_msgs__msg__SystemStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__SystemStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__SystemStatus__Sequence__copy(
  const eufs_msgs__msg__SystemStatus__Sequence * input,
  eufs_msgs__msg__SystemStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__SystemStatus);
    eufs_msgs__msg__SystemStatus * data =
      (eufs_msgs__msg__SystemStatus *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__SystemStatus__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__SystemStatus__fini(&data[i]);
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
    if (!eufs_msgs__msg__SystemStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
