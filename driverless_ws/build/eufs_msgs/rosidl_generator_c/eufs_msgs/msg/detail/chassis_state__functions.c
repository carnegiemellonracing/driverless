// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/ChassisState.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/chassis_state__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `steering_commander`
// Member `throttle_commander`
// Member `front_brake_commander`
#include "rosidl_runtime_c/string_functions.h"

bool
eufs_msgs__msg__ChassisState__init(eufs_msgs__msg__ChassisState * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    eufs_msgs__msg__ChassisState__fini(msg);
    return false;
  }
  // throttle_relay_enabled
  // autonomous_enabled
  // runstop_motion_enabled
  // steering_commander
  if (!rosidl_runtime_c__String__init(&msg->steering_commander)) {
    eufs_msgs__msg__ChassisState__fini(msg);
    return false;
  }
  // steering
  // throttle_commander
  if (!rosidl_runtime_c__String__init(&msg->throttle_commander)) {
    eufs_msgs__msg__ChassisState__fini(msg);
    return false;
  }
  // throttle
  // front_brake_commander
  if (!rosidl_runtime_c__String__init(&msg->front_brake_commander)) {
    eufs_msgs__msg__ChassisState__fini(msg);
    return false;
  }
  // front_brake
  return true;
}

void
eufs_msgs__msg__ChassisState__fini(eufs_msgs__msg__ChassisState * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // throttle_relay_enabled
  // autonomous_enabled
  // runstop_motion_enabled
  // steering_commander
  rosidl_runtime_c__String__fini(&msg->steering_commander);
  // steering
  // throttle_commander
  rosidl_runtime_c__String__fini(&msg->throttle_commander);
  // throttle
  // front_brake_commander
  rosidl_runtime_c__String__fini(&msg->front_brake_commander);
  // front_brake
}

bool
eufs_msgs__msg__ChassisState__are_equal(const eufs_msgs__msg__ChassisState * lhs, const eufs_msgs__msg__ChassisState * rhs)
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
  // throttle_relay_enabled
  if (lhs->throttle_relay_enabled != rhs->throttle_relay_enabled) {
    return false;
  }
  // autonomous_enabled
  if (lhs->autonomous_enabled != rhs->autonomous_enabled) {
    return false;
  }
  // runstop_motion_enabled
  if (lhs->runstop_motion_enabled != rhs->runstop_motion_enabled) {
    return false;
  }
  // steering_commander
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->steering_commander), &(rhs->steering_commander)))
  {
    return false;
  }
  // steering
  if (lhs->steering != rhs->steering) {
    return false;
  }
  // throttle_commander
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->throttle_commander), &(rhs->throttle_commander)))
  {
    return false;
  }
  // throttle
  if (lhs->throttle != rhs->throttle) {
    return false;
  }
  // front_brake_commander
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->front_brake_commander), &(rhs->front_brake_commander)))
  {
    return false;
  }
  // front_brake
  if (lhs->front_brake != rhs->front_brake) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__ChassisState__copy(
  const eufs_msgs__msg__ChassisState * input,
  eufs_msgs__msg__ChassisState * output)
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
  // throttle_relay_enabled
  output->throttle_relay_enabled = input->throttle_relay_enabled;
  // autonomous_enabled
  output->autonomous_enabled = input->autonomous_enabled;
  // runstop_motion_enabled
  output->runstop_motion_enabled = input->runstop_motion_enabled;
  // steering_commander
  if (!rosidl_runtime_c__String__copy(
      &(input->steering_commander), &(output->steering_commander)))
  {
    return false;
  }
  // steering
  output->steering = input->steering;
  // throttle_commander
  if (!rosidl_runtime_c__String__copy(
      &(input->throttle_commander), &(output->throttle_commander)))
  {
    return false;
  }
  // throttle
  output->throttle = input->throttle;
  // front_brake_commander
  if (!rosidl_runtime_c__String__copy(
      &(input->front_brake_commander), &(output->front_brake_commander)))
  {
    return false;
  }
  // front_brake
  output->front_brake = input->front_brake;
  return true;
}

eufs_msgs__msg__ChassisState *
eufs_msgs__msg__ChassisState__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__ChassisState * msg = (eufs_msgs__msg__ChassisState *)allocator.allocate(sizeof(eufs_msgs__msg__ChassisState), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__ChassisState));
  bool success = eufs_msgs__msg__ChassisState__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__ChassisState__destroy(eufs_msgs__msg__ChassisState * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__ChassisState__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__ChassisState__Sequence__init(eufs_msgs__msg__ChassisState__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__ChassisState * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__ChassisState *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__ChassisState), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__ChassisState__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__ChassisState__fini(&data[i - 1]);
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
eufs_msgs__msg__ChassisState__Sequence__fini(eufs_msgs__msg__ChassisState__Sequence * array)
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
      eufs_msgs__msg__ChassisState__fini(&array->data[i]);
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

eufs_msgs__msg__ChassisState__Sequence *
eufs_msgs__msg__ChassisState__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__ChassisState__Sequence * array = (eufs_msgs__msg__ChassisState__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__ChassisState__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__ChassisState__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__ChassisState__Sequence__destroy(eufs_msgs__msg__ChassisState__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__ChassisState__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__ChassisState__Sequence__are_equal(const eufs_msgs__msg__ChassisState__Sequence * lhs, const eufs_msgs__msg__ChassisState__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__ChassisState__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__ChassisState__Sequence__copy(
  const eufs_msgs__msg__ChassisState__Sequence * input,
  eufs_msgs__msg__ChassisState__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__ChassisState);
    eufs_msgs__msg__ChassisState * data =
      (eufs_msgs__msg__ChassisState *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__ChassisState__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__ChassisState__fini(&data[i]);
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
    if (!eufs_msgs__msg__ChassisState__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
