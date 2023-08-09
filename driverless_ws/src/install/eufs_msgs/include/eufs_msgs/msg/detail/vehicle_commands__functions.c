// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:msg/VehicleCommands.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/msg/detail/vehicle_commands__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


bool
eufs_msgs__msg__VehicleCommands__init(eufs_msgs__msg__VehicleCommands * msg)
{
  if (!msg) {
    return false;
  }
  // handshake
  // ebs
  // direction
  // mission_status
  // braking
  // torque
  // steering
  // rpm
  return true;
}

void
eufs_msgs__msg__VehicleCommands__fini(eufs_msgs__msg__VehicleCommands * msg)
{
  if (!msg) {
    return;
  }
  // handshake
  // ebs
  // direction
  // mission_status
  // braking
  // torque
  // steering
  // rpm
}

bool
eufs_msgs__msg__VehicleCommands__are_equal(const eufs_msgs__msg__VehicleCommands * lhs, const eufs_msgs__msg__VehicleCommands * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // handshake
  if (lhs->handshake != rhs->handshake) {
    return false;
  }
  // ebs
  if (lhs->ebs != rhs->ebs) {
    return false;
  }
  // direction
  if (lhs->direction != rhs->direction) {
    return false;
  }
  // mission_status
  if (lhs->mission_status != rhs->mission_status) {
    return false;
  }
  // braking
  if (lhs->braking != rhs->braking) {
    return false;
  }
  // torque
  if (lhs->torque != rhs->torque) {
    return false;
  }
  // steering
  if (lhs->steering != rhs->steering) {
    return false;
  }
  // rpm
  if (lhs->rpm != rhs->rpm) {
    return false;
  }
  return true;
}

bool
eufs_msgs__msg__VehicleCommands__copy(
  const eufs_msgs__msg__VehicleCommands * input,
  eufs_msgs__msg__VehicleCommands * output)
{
  if (!input || !output) {
    return false;
  }
  // handshake
  output->handshake = input->handshake;
  // ebs
  output->ebs = input->ebs;
  // direction
  output->direction = input->direction;
  // mission_status
  output->mission_status = input->mission_status;
  // braking
  output->braking = input->braking;
  // torque
  output->torque = input->torque;
  // steering
  output->steering = input->steering;
  // rpm
  output->rpm = input->rpm;
  return true;
}

eufs_msgs__msg__VehicleCommands *
eufs_msgs__msg__VehicleCommands__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__VehicleCommands * msg = (eufs_msgs__msg__VehicleCommands *)allocator.allocate(sizeof(eufs_msgs__msg__VehicleCommands), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__msg__VehicleCommands));
  bool success = eufs_msgs__msg__VehicleCommands__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__msg__VehicleCommands__destroy(eufs_msgs__msg__VehicleCommands * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__msg__VehicleCommands__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__msg__VehicleCommands__Sequence__init(eufs_msgs__msg__VehicleCommands__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__VehicleCommands * data = NULL;

  if (size) {
    data = (eufs_msgs__msg__VehicleCommands *)allocator.zero_allocate(size, sizeof(eufs_msgs__msg__VehicleCommands), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__msg__VehicleCommands__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__msg__VehicleCommands__fini(&data[i - 1]);
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
eufs_msgs__msg__VehicleCommands__Sequence__fini(eufs_msgs__msg__VehicleCommands__Sequence * array)
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
      eufs_msgs__msg__VehicleCommands__fini(&array->data[i]);
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

eufs_msgs__msg__VehicleCommands__Sequence *
eufs_msgs__msg__VehicleCommands__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__msg__VehicleCommands__Sequence * array = (eufs_msgs__msg__VehicleCommands__Sequence *)allocator.allocate(sizeof(eufs_msgs__msg__VehicleCommands__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__msg__VehicleCommands__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__msg__VehicleCommands__Sequence__destroy(eufs_msgs__msg__VehicleCommands__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__msg__VehicleCommands__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__msg__VehicleCommands__Sequence__are_equal(const eufs_msgs__msg__VehicleCommands__Sequence * lhs, const eufs_msgs__msg__VehicleCommands__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__msg__VehicleCommands__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__msg__VehicleCommands__Sequence__copy(
  const eufs_msgs__msg__VehicleCommands__Sequence * input,
  eufs_msgs__msg__VehicleCommands__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__msg__VehicleCommands);
    eufs_msgs__msg__VehicleCommands * data =
      (eufs_msgs__msg__VehicleCommands *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__msg__VehicleCommands__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__msg__VehicleCommands__fini(&data[i]);
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
    if (!eufs_msgs__msg__VehicleCommands__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
