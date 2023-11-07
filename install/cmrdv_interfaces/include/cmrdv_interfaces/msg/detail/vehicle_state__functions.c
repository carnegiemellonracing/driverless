// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from cmrdv_interfaces:msg/VehicleState.idl
// generated code does not contain a copyright notice
#include "cmrdv_interfaces/msg/detail/vehicle_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `position`
#include "geometry_msgs/msg/detail/pose__functions.h"
// Member `velocity`
#include "geometry_msgs/msg/detail/twist__functions.h"
// Member `acceleration`
#include "geometry_msgs/msg/detail/accel__functions.h"

bool
cmrdv_interfaces__msg__VehicleState__init(cmrdv_interfaces__msg__VehicleState * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    cmrdv_interfaces__msg__VehicleState__fini(msg);
    return false;
  }
  // position
  if (!geometry_msgs__msg__Pose__init(&msg->position)) {
    cmrdv_interfaces__msg__VehicleState__fini(msg);
    return false;
  }
  // velocity
  if (!geometry_msgs__msg__Twist__init(&msg->velocity)) {
    cmrdv_interfaces__msg__VehicleState__fini(msg);
    return false;
  }
  // acceleration
  if (!geometry_msgs__msg__Accel__init(&msg->acceleration)) {
    cmrdv_interfaces__msg__VehicleState__fini(msg);
    return false;
  }
  return true;
}

void
cmrdv_interfaces__msg__VehicleState__fini(cmrdv_interfaces__msg__VehicleState * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // position
  geometry_msgs__msg__Pose__fini(&msg->position);
  // velocity
  geometry_msgs__msg__Twist__fini(&msg->velocity);
  // acceleration
  geometry_msgs__msg__Accel__fini(&msg->acceleration);
}

bool
cmrdv_interfaces__msg__VehicleState__are_equal(const cmrdv_interfaces__msg__VehicleState * lhs, const cmrdv_interfaces__msg__VehicleState * rhs)
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
  // position
  if (!geometry_msgs__msg__Pose__are_equal(
      &(lhs->position), &(rhs->position)))
  {
    return false;
  }
  // velocity
  if (!geometry_msgs__msg__Twist__are_equal(
      &(lhs->velocity), &(rhs->velocity)))
  {
    return false;
  }
  // acceleration
  if (!geometry_msgs__msg__Accel__are_equal(
      &(lhs->acceleration), &(rhs->acceleration)))
  {
    return false;
  }
  return true;
}

bool
cmrdv_interfaces__msg__VehicleState__copy(
  const cmrdv_interfaces__msg__VehicleState * input,
  cmrdv_interfaces__msg__VehicleState * output)
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
  // position
  if (!geometry_msgs__msg__Pose__copy(
      &(input->position), &(output->position)))
  {
    return false;
  }
  // velocity
  if (!geometry_msgs__msg__Twist__copy(
      &(input->velocity), &(output->velocity)))
  {
    return false;
  }
  // acceleration
  if (!geometry_msgs__msg__Accel__copy(
      &(input->acceleration), &(output->acceleration)))
  {
    return false;
  }
  return true;
}

cmrdv_interfaces__msg__VehicleState *
cmrdv_interfaces__msg__VehicleState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__VehicleState * msg = (cmrdv_interfaces__msg__VehicleState *)allocator.allocate(sizeof(cmrdv_interfaces__msg__VehicleState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(cmrdv_interfaces__msg__VehicleState));
  bool success = cmrdv_interfaces__msg__VehicleState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
cmrdv_interfaces__msg__VehicleState__destroy(cmrdv_interfaces__msg__VehicleState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    cmrdv_interfaces__msg__VehicleState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
cmrdv_interfaces__msg__VehicleState__Sequence__init(cmrdv_interfaces__msg__VehicleState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__VehicleState * data = NULL;

  if (size) {
    data = (cmrdv_interfaces__msg__VehicleState *)allocator.zero_allocate(size, sizeof(cmrdv_interfaces__msg__VehicleState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = cmrdv_interfaces__msg__VehicleState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        cmrdv_interfaces__msg__VehicleState__fini(&data[i - 1]);
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
cmrdv_interfaces__msg__VehicleState__Sequence__fini(cmrdv_interfaces__msg__VehicleState__Sequence * array)
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
      cmrdv_interfaces__msg__VehicleState__fini(&array->data[i]);
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

cmrdv_interfaces__msg__VehicleState__Sequence *
cmrdv_interfaces__msg__VehicleState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  cmrdv_interfaces__msg__VehicleState__Sequence * array = (cmrdv_interfaces__msg__VehicleState__Sequence *)allocator.allocate(sizeof(cmrdv_interfaces__msg__VehicleState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = cmrdv_interfaces__msg__VehicleState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
cmrdv_interfaces__msg__VehicleState__Sequence__destroy(cmrdv_interfaces__msg__VehicleState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    cmrdv_interfaces__msg__VehicleState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
cmrdv_interfaces__msg__VehicleState__Sequence__are_equal(const cmrdv_interfaces__msg__VehicleState__Sequence * lhs, const cmrdv_interfaces__msg__VehicleState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!cmrdv_interfaces__msg__VehicleState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
cmrdv_interfaces__msg__VehicleState__Sequence__copy(
  const cmrdv_interfaces__msg__VehicleState__Sequence * input,
  cmrdv_interfaces__msg__VehicleState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(cmrdv_interfaces__msg__VehicleState);
    cmrdv_interfaces__msg__VehicleState * data =
      (cmrdv_interfaces__msg__VehicleState *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!cmrdv_interfaces__msg__VehicleState__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          cmrdv_interfaces__msg__VehicleState__fini(&data[i]);
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
    if (!cmrdv_interfaces__msg__VehicleState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
