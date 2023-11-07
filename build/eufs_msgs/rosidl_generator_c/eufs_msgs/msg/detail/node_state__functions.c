// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/NodeState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/node_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `name`
#include "rosidl_runtime_c/string_functions.h"
// Member `last_heartbeat`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
eufs_msgs__msg__NodeState__init(eufs_msgs__msg__NodeState * msg)
{
  if (!msg) {
    return false;
  }
  // id
  // name
  if (!rosidl_runtime_c__String__init(&msg->name)) {
    eufs_msgs__msg__NodeState__fini(msg);
    return false;
  }
  // exp_heartbeat
  // last_heartbeat
  if (!builtin_interfaces__msg__Time__init(&msg->last_heartbeat)) {
    eufs_msgs__msg__NodeState__fini(msg);
    return false;
  }
  // severity
  // online
  return true;
}

void
eufs_msgs__msg__NodeState__fini(eufs_msgs__msg__NodeState * msg)
{
  if (!msg) {
    return;
  }
  // id
  // name
  rosidl_runtime_c__String__fini(&msg->name);
  // exp_heartbeat
  // last_heartbeat
  builtin_interfaces__msg__Time__fini(&msg->last_heartbeat);
  // severity
  // online
}

bool
eufs_msgs__msg__NodeState__are_equal(const eufs_msgs__msg__NodeState * lhs, const eufs_msgs__msg__NodeState * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  // name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->name), &(rhs->name)))
  {
    return false;
  }
  // exp_heartbeat
  if (lhs->exp_heartbeat != rhs->exp_heartbeat) {
    return false;
  }
  // last_heartbeat
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->last_heartbeat), &(rhs->last_heartbeat)))
  {
    return false;
  }
  // severity
  if (lhs->severity != rhs->severity) {
    return false;
  }
  // online
  if (lhs->online != rhs->online) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__NodeState__copy(
  const eufs_msgs__msg__NodeState * input,
  eufs_msgs__msg__NodeState * output)
{
  if (!input || !output) {
    return false;
  }
  // id
  output->id = input->id;
  // name
  if (!rosidl_runtime_c__String__copy(
      &(input->name), &(output->name)))
  {
    return false;
  }
  // exp_heartbeat
  output->exp_heartbeat = input->exp_heartbeat;
  // last_heartbeat
  if (!builtin_interfaces__msg__Time__copy(
      &(input->last_heartbeat), &(output->last_heartbeat)))
  {
    return false;
  }
  // severity
  output->severity = input->severity;
  // online
  output->online = input->online;
  return true;
}

eufs_msgs__msg__NodeState *
eufs_msgs__msg__NodeState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__NodeState * msg = (eufs_msgs__msg__NodeState *)allocator.allocate(sizeof(eufs_msgs__msg__NodeState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__NodeState));
  bool success = eufs_msgs__msg__NodeState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__NodeState__destroy(eufs_msgs__msg__NodeState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__NodeState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__NodeState__Sequence__init(eufs_msgs__msg__NodeState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__NodeState * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__NodeState *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__NodeState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__NodeState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__NodeState__fini(&data[i - 1]);
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
eufs_msgs__msg__NodeState__Sequence__fini(eufs_msgs__msg__NodeState__Sequence * array)
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
      eufs_msgs__msg__NodeState__fini(&array->data[i]);
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

eufs_msgs__msg__NodeState__Sequence *
eufs_msgs__msg__NodeState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__NodeState__Sequence * array = (eufs_msgs__msg__NodeState__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__NodeState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__NodeState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__NodeState__Sequence__destroy(eufs_msgs__msg__NodeState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__NodeState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__NodeState__Sequence__are_equal(const eufs_msgs__msg__NodeState__Sequence * lhs, const eufs_msgs__msg__NodeState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__NodeState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__NodeState__Sequence__copy(
  const eufs_msgs__msg__NodeState__Sequence * input,
  eufs_msgs__msg__NodeState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__NodeState);
    eufs_msgs__msg__NodeState * data =
      (eufs_msgs__msg__NodeState *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__NodeState__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__NodeState__fini(&data[i]);
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
    if (!eufs_msgs__msg__NodeState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
