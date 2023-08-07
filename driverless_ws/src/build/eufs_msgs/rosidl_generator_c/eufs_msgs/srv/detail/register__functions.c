// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:srv/Register.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/srv/detail/register__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"

// Include directives for member types
// Member `node_name`
#include "rosidl_runtime_c/string_functions.h"

bool
eufs_msgs__srv__Register_Request__init(eufs_msgs__srv__Register_Request * msg)
{
  if (!msg) {
    return false;
  }
  // node_name
  if (!rosidl_runtime_c__String__init(&msg->node_name)) {
    eufs_msgs__srv__Register_Request__fini(msg);
    return false;
  }
  // severity
  return true;
}

void
eufs_msgs__srv__Register_Request__fini(eufs_msgs__srv__Register_Request * msg)
{
  if (!msg) {
    return;
  }
  // node_name
  rosidl_runtime_c__String__fini(&msg->node_name);
  // severity
}

bool
eufs_msgs__srv__Register_Request__are_equal(const eufs_msgs__srv__Register_Request * lhs, const eufs_msgs__srv__Register_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // node_name
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->node_name), &(rhs->node_name)))
  {
    return false;
  }
  // severity
  if (lhs->severity != rhs->severity) {
    return false;
  }
  return true;
}

bool
eufs_msgs__srv__Register_Request__copy(
  const eufs_msgs__srv__Register_Request * input,
  eufs_msgs__srv__Register_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // node_name
  if (!rosidl_runtime_c__String__copy(
      &(input->node_name), &(output->node_name)))
  {
    return false;
  }
  // severity
  output->severity = input->severity;
  return true;
}

eufs_msgs__srv__Register_Request *
eufs_msgs__srv__Register_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__srv__Register_Request * msg = (eufs_msgs__srv__Register_Request *)allocator.allocate(sizeof(eufs_msgs__srv__Register_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__srv__Register_Request));
  bool success = eufs_msgs__srv__Register_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__srv__Register_Request__destroy(eufs_msgs__srv__Register_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__srv__Register_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__srv__Register_Request__Sequence__init(eufs_msgs__srv__Register_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__srv__Register_Request * data = NULL;

  if (size) {
    data = (eufs_msgs__srv__Register_Request *)allocator.zero_allocate(size, sizeof(eufs_msgs__srv__Register_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__srv__Register_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__srv__Register_Request__fini(&data[i - 1]);
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
eufs_msgs__srv__Register_Request__Sequence__fini(eufs_msgs__srv__Register_Request__Sequence * array)
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
      eufs_msgs__srv__Register_Request__fini(&array->data[i]);
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

eufs_msgs__srv__Register_Request__Sequence *
eufs_msgs__srv__Register_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__srv__Register_Request__Sequence * array = (eufs_msgs__srv__Register_Request__Sequence *)allocator.allocate(sizeof(eufs_msgs__srv__Register_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__srv__Register_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__srv__Register_Request__Sequence__destroy(eufs_msgs__srv__Register_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__srv__Register_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__srv__Register_Request__Sequence__are_equal(const eufs_msgs__srv__Register_Request__Sequence * lhs, const eufs_msgs__srv__Register_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__srv__Register_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__srv__Register_Request__Sequence__copy(
  const eufs_msgs__srv__Register_Request__Sequence * input,
  eufs_msgs__srv__Register_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__srv__Register_Request);
    eufs_msgs__srv__Register_Request * data =
      (eufs_msgs__srv__Register_Request *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__srv__Register_Request__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__srv__Register_Request__fini(&data[i]);
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
    if (!eufs_msgs__srv__Register_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
eufs_msgs__srv__Register_Response__init(eufs_msgs__srv__Register_Response * msg)
{
  if (!msg) {
    return false;
  }
  // id
  return true;
}

void
eufs_msgs__srv__Register_Response__fini(eufs_msgs__srv__Register_Response * msg)
{
  if (!msg) {
    return;
  }
  // id
}

bool
eufs_msgs__srv__Register_Response__are_equal(const eufs_msgs__srv__Register_Response * lhs, const eufs_msgs__srv__Register_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  return true;
}

bool
eufs_msgs__srv__Register_Response__copy(
  const eufs_msgs__srv__Register_Response * input,
  eufs_msgs__srv__Register_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // id
  output->id = input->id;
  return true;
}

eufs_msgs__srv__Register_Response *
eufs_msgs__srv__Register_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__srv__Register_Response * msg = (eufs_msgs__srv__Register_Response *)allocator.allocate(sizeof(eufs_msgs__srv__Register_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__srv__Register_Response));
  bool success = eufs_msgs__srv__Register_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__srv__Register_Response__destroy(eufs_msgs__srv__Register_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__srv__Register_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__srv__Register_Response__Sequence__init(eufs_msgs__srv__Register_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__srv__Register_Response * data = NULL;

  if (size) {
    data = (eufs_msgs__srv__Register_Response *)allocator.zero_allocate(size, sizeof(eufs_msgs__srv__Register_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__srv__Register_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__srv__Register_Response__fini(&data[i - 1]);
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
eufs_msgs__srv__Register_Response__Sequence__fini(eufs_msgs__srv__Register_Response__Sequence * array)
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
      eufs_msgs__srv__Register_Response__fini(&array->data[i]);
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

eufs_msgs__srv__Register_Response__Sequence *
eufs_msgs__srv__Register_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__srv__Register_Response__Sequence * array = (eufs_msgs__srv__Register_Response__Sequence *)allocator.allocate(sizeof(eufs_msgs__srv__Register_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__srv__Register_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__srv__Register_Response__Sequence__destroy(eufs_msgs__srv__Register_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__srv__Register_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__srv__Register_Response__Sequence__are_equal(const eufs_msgs__srv__Register_Response__Sequence * lhs, const eufs_msgs__srv__Register_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__srv__Register_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__srv__Register_Response__Sequence__copy(
  const eufs_msgs__srv__Register_Response__Sequence * input,
  eufs_msgs__srv__Register_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__srv__Register_Response);
    eufs_msgs__srv__Register_Response * data =
      (eufs_msgs__srv__Register_Response *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__srv__Register_Response__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__srv__Register_Response__fini(&data[i]);
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
    if (!eufs_msgs__srv__Register_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
