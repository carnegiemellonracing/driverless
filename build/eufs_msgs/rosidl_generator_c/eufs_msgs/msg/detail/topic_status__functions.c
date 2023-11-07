// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/TopicStatus.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/topic_status__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `topic`
// Member `description`
// Member `group`
// Member `log_level`
#include "rosidl_runtime_c/string_functions.h"

bool
eufs_msgs__msg__TopicStatus__init(eufs_msgs__msg__TopicStatus * msg)
{
  if (!msg) {
    return false;
  }
  // topic
  if (!rosidl_runtime_c__String__init(&msg->topic)) {
    eufs_msgs__msg__TopicStatus__fini(msg);
    return false;
  }
  // description
  if (!rosidl_runtime_c__String__init(&msg->description)) {
    eufs_msgs__msg__TopicStatus__fini(msg);
    return false;
  }
  // group
  if (!rosidl_runtime_c__String__init(&msg->group)) {
    eufs_msgs__msg__TopicStatus__fini(msg);
    return false;
  }
  // trigger_ebs
  // log_level
  if (!rosidl_runtime_c__String__init(&msg->log_level)) {
    eufs_msgs__msg__TopicStatus__fini(msg);
    return false;
  }
  // status
  return true;
}

void
eufs_msgs__msg__TopicStatus__fini(eufs_msgs__msg__TopicStatus * msg)
{
  if (!msg) {
    return;
  }
  // topic
  rosidl_runtime_c__String__fini(&msg->topic);
  // description
  rosidl_runtime_c__String__fini(&msg->description);
  // group
  rosidl_runtime_c__String__fini(&msg->group);
  // trigger_ebs
  // log_level
  rosidl_runtime_c__String__fini(&msg->log_level);
  // status
}

bool
eufs_msgs__msg__TopicStatus__are_equal(const eufs_msgs__msg__TopicStatus * lhs, const eufs_msgs__msg__TopicStatus * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // topic
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->topic), &(rhs->topic)))
  {
    return false;
  }
  // description
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->description), &(rhs->description)))
  {
    return false;
  }
  // group
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->group), &(rhs->group)))
  {
    return false;
  }
  // trigger_ebs
  if (lhs->trigger_ebs != rhs->trigger_ebs) {
    return false;
  }
  // log_level
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->log_level), &(rhs->log_level)))
  {
    return false;
  }
  // status
  if (lhs->status != rhs->status) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__TopicStatus__copy(
  const eufs_msgs__msg__TopicStatus * input,
  eufs_msgs__msg__TopicStatus * output)
{
  if (!input || !output) {
    return false;
  }
  // topic
  if (!rosidl_runtime_c__String__copy(
      &(input->topic), &(output->topic)))
  {
    return false;
  }
  // description
  if (!rosidl_runtime_c__String__copy(
      &(input->description), &(output->description)))
  {
    return false;
  }
  // group
  if (!rosidl_runtime_c__String__copy(
      &(input->group), &(output->group)))
  {
    return false;
  }
  // trigger_ebs
  output->trigger_ebs = input->trigger_ebs;
  // log_level
  if (!rosidl_runtime_c__String__copy(
      &(input->log_level), &(output->log_level)))
  {
    return false;
  }
  // status
  output->status = input->status;
  return true;
}

eufs_msgs__msg__TopicStatus *
eufs_msgs__msg__TopicStatus__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__TopicStatus * msg = (eufs_msgs__msg__TopicStatus *)allocator.allocate(sizeof(eufs_msgs__msg__TopicStatus), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__TopicStatus));
  bool success = eufs_msgs__msg__TopicStatus__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__TopicStatus__destroy(eufs_msgs__msg__TopicStatus * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__TopicStatus__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__TopicStatus__Sequence__init(eufs_msgs__msg__TopicStatus__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__TopicStatus * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__TopicStatus *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__TopicStatus), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__TopicStatus__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__TopicStatus__fini(&data[i - 1]);
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
eufs_msgs__msg__TopicStatus__Sequence__fini(eufs_msgs__msg__TopicStatus__Sequence * array)
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
      eufs_msgs__msg__TopicStatus__fini(&array->data[i]);
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

eufs_msgs__msg__TopicStatus__Sequence *
eufs_msgs__msg__TopicStatus__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__TopicStatus__Sequence * array = (eufs_msgs__msg__TopicStatus__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__TopicStatus__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__TopicStatus__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__TopicStatus__Sequence__destroy(eufs_msgs__msg__TopicStatus__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__TopicStatus__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__TopicStatus__Sequence__are_equal(const eufs_msgs__msg__TopicStatus__Sequence * lhs, const eufs_msgs__msg__TopicStatus__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__TopicStatus__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__TopicStatus__Sequence__copy(
  const eufs_msgs__msg__TopicStatus__Sequence * input,
  eufs_msgs__msg__TopicStatus__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__TopicStatus);
    eufs_msgs__msg__TopicStatus * data =
      (eufs_msgs__msg__TopicStatus *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__TopicStatus__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__TopicStatus__fini(&data[i]);
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
    if (!eufs_msgs__msg__TopicStatus__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
