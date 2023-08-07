// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/PurePursuitCheckpointArrayStamped.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/pure_pursuit_checkpoint_array_stamped__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `checkpoints`
#include "eufs_msgs/msg/detail/pure_pursuit_checkpoint__functions.h"

bool
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__init(eufs_msgs__msg__PurePursuitCheckpointArrayStamped * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__PurePursuitCheckpointArrayStamped__fini(msg);
    return false;
  }
  // checkpoints
  if (!eufs_msgs__msg__PurePursuitCheckpoint__Sequence__init(&msg->checkpoints, 0)) {
    eufs_msgs__msg__PurePursuitCheckpointArrayStamped__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__fini(eufs_msgs__msg__PurePursuitCheckpointArrayStamped * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // checkpoints
  eufs_msgs__msg__PurePursuitCheckpoint__Sequence__fini(&msg->checkpoints);
}

bool
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__are_equal(const eufs_msgs__msg__PurePursuitCheckpointArrayStamped * lhs, const eufs_msgs__msg__PurePursuitCheckpointArrayStamped * rhs)
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
  // checkpoints
  if (!eufs_msgs__msg__PurePursuitCheckpoint__Sequence__are_equal(
      &(lhs->checkpoints), &(rhs->checkpoints)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__copy(
  const eufs_msgs__msg__PurePursuitCheckpointArrayStamped * input,
  eufs_msgs__msg__PurePursuitCheckpointArrayStamped * output)
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
  // checkpoints
  if (!eufs_msgs__msg__PurePursuitCheckpoint__Sequence__copy(
      &(input->checkpoints), &(output->checkpoints)))
  {
    return false;
  }
  return true;
}

eufs_msgs__msg__PurePursuitCheckpointArrayStamped *
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PurePursuitCheckpointArrayStamped * msg = (eufs_msgs__msg__PurePursuitCheckpointArrayStamped *)allocator.allocate(sizeof(eufs_msgs__msg__PurePursuitCheckpointArrayStamped), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__PurePursuitCheckpointArrayStamped));
  bool success = eufs_msgs__msg__PurePursuitCheckpointArrayStamped__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__destroy(eufs_msgs__msg__PurePursuitCheckpointArrayStamped * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__PurePursuitCheckpointArrayStamped__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence__init(eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PurePursuitCheckpointArrayStamped * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__PurePursuitCheckpointArrayStamped *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__PurePursuitCheckpointArrayStamped), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__PurePursuitCheckpointArrayStamped__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__PurePursuitCheckpointArrayStamped__fini(&data[i - 1]);
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
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence__fini(eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence * array)
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
      eufs_msgs__msg__PurePursuitCheckpointArrayStamped__fini(&array->data[i]);
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

eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence *
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence * array = (eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence__destroy(eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence__are_equal(const eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence * lhs, const eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__PurePursuitCheckpointArrayStamped__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence__copy(
  const eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence * input,
  eufs_msgs__msg__PurePursuitCheckpointArrayStamped__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__PurePursuitCheckpointArrayStamped);
    eufs_msgs__msg__PurePursuitCheckpointArrayStamped * data =
      (eufs_msgs__msg__PurePursuitCheckpointArrayStamped *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__PurePursuitCheckpointArrayStamped__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__PurePursuitCheckpointArrayStamped__fini(&data[i]);
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
    if (!eufs_msgs__msg__PurePursuitCheckpointArrayStamped__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
