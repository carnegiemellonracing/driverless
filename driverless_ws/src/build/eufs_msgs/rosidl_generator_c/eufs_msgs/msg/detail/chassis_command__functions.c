// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/ChassisCommand.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/chassis_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `sender`
#include "rosidl_runtime_c/string_functions.h"

bool
eufs_msgs__msg__ChassisCommand__init(eufs_msgs__msg__ChassisCommand * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__ChassisCommand__fini(msg);
    return false;
  }
  // sender
  if (!rosidl_runtime_c__String__init(&msg->sender)) {
    eufs_msgs__msg__ChassisCommand__fini(msg);
    return false;
  }
  // throttle
  // steering
  // front_brake
  return true;
}

void
eufs_msgs__msg__ChassisCommand__fini(eufs_msgs__msg__ChassisCommand * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // sender
  rosidl_runtime_c__String__fini(&msg->sender);
  // throttle
  // steering
  // front_brake
}

bool
eufs_msgs__msg__ChassisCommand__are_equal(const eufs_msgs__msg__ChassisCommand * lhs, const eufs_msgs__msg__ChassisCommand * rhs)
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
  // sender
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->sender), &(rhs->sender)))
  {
    return false;
  }
  // throttle
  if (lhs->throttle != rhs->throttle) {
    return false;
  }
  // steering
  if (lhs->steering != rhs->steering) {
    return false;
  }
  // front_brake
  if (lhs->front_brake != rhs->front_brake) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__ChassisCommand__copy(
  const eufs_msgs__msg__ChassisCommand * input,
  eufs_msgs__msg__ChassisCommand * output)
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
  // sender
  if (!rosidl_runtime_c__String__copy(
      &(input->sender), &(output->sender)))
  {
    return false;
  }
  // throttle
  output->throttle = input->throttle;
  // steering
  output->steering = input->steering;
  // front_brake
  output->front_brake = input->front_brake;
  return true;
}

eufs_msgs__msg__ChassisCommand *
eufs_msgs__msg__ChassisCommand__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__ChassisCommand * msg = (eufs_msgs__msg__ChassisCommand *)allocator.allocate(sizeof(eufs_msgs__msg__ChassisCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__ChassisCommand));
  bool success = eufs_msgs__msg__ChassisCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__ChassisCommand__destroy(eufs_msgs__msg__ChassisCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__ChassisCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__ChassisCommand__Sequence__init(eufs_msgs__msg__ChassisCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__ChassisCommand * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__ChassisCommand *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__ChassisCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__ChassisCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__ChassisCommand__fini(&data[i - 1]);
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
eufs_msgs__msg__ChassisCommand__Sequence__fini(eufs_msgs__msg__ChassisCommand__Sequence * array)
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
      eufs_msgs__msg__ChassisCommand__fini(&array->data[i]);
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

eufs_msgs__msg__ChassisCommand__Sequence *
eufs_msgs__msg__ChassisCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__ChassisCommand__Sequence * array = (eufs_msgs__msg__ChassisCommand__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__ChassisCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__ChassisCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__ChassisCommand__Sequence__destroy(eufs_msgs__msg__ChassisCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__ChassisCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__ChassisCommand__Sequence__are_equal(const eufs_msgs__msg__ChassisCommand__Sequence * lhs, const eufs_msgs__msg__ChassisCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__ChassisCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__ChassisCommand__Sequence__copy(
  const eufs_msgs__msg__ChassisCommand__Sequence * input,
  eufs_msgs__msg__ChassisCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__ChassisCommand);
    eufs_msgs__msg__ChassisCommand * data =
      (eufs_msgs__msg__ChassisCommand *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__ChassisCommand__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__ChassisCommand__fini(&data[i]);
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
    if (!eufs_msgs__msg__ChassisCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
