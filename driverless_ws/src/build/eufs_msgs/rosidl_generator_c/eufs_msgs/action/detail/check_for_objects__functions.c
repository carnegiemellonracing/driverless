// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from eufs_msgs:action/CheckForObjects.idl
// generated code does not contain a copyright notice
#include "eufs_msgs/action/detail/check_for_objects__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `image`
#include "sensor_msgs/msg/detail/image__functions.h"

bool
eufs_msgs__action__CheckForObjects_Goal__init(eufs_msgs__action__CheckForObjects_Goal * msg)
{
  if (!msg) {
    return false;
  }
  // id
  // image
  if (!sensor_msgs__msg__Image__init(&msg->image)) {
    eufs_msgs__action__CheckForObjects_Goal__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__action__CheckForObjects_Goal__fini(eufs_msgs__action__CheckForObjects_Goal * msg)
{
  if (!msg) {
    return;
  }
  // id
  // image
  sensor_msgs__msg__Image__fini(&msg->image);
}

bool
eufs_msgs__action__CheckForObjects_Goal__are_equal(const eufs_msgs__action__CheckForObjects_Goal * lhs, const eufs_msgs__action__CheckForObjects_Goal * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  // image
  if (!sensor_msgs__msg__Image__are_equal(
      &(lhs->image), &(rhs->image)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_Goal__copy(
  const eufs_msgs__action__CheckForObjects_Goal * input,
  eufs_msgs__action__CheckForObjects_Goal * output)
{
  if (!input || !output) {
    return false;
  }
  // id
  output->id = input->id;
  // image
  if (!sensor_msgs__msg__Image__copy(
      &(input->image), &(output->image)))
  {
    return false;
  }
  return true;
}

eufs_msgs__action__CheckForObjects_Goal *
eufs_msgs__action__CheckForObjects_Goal__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_Goal * msg = (eufs_msgs__action__CheckForObjects_Goal *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_Goal), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__action__CheckForObjects_Goal));
  bool success = eufs_msgs__action__CheckForObjects_Goal__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__action__CheckForObjects_Goal__destroy(eufs_msgs__action__CheckForObjects_Goal * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__action__CheckForObjects_Goal__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__action__CheckForObjects_Goal__Sequence__init(eufs_msgs__action__CheckForObjects_Goal__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_Goal * data = NULL;

  if (size) {
    data = (eufs_msgs__action__CheckForObjects_Goal *)allocator.zero_allocate(size, sizeof(eufs_msgs__action__CheckForObjects_Goal), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__action__CheckForObjects_Goal__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__action__CheckForObjects_Goal__fini(&data[i - 1]);
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
eufs_msgs__action__CheckForObjects_Goal__Sequence__fini(eufs_msgs__action__CheckForObjects_Goal__Sequence * array)
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
      eufs_msgs__action__CheckForObjects_Goal__fini(&array->data[i]);
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

eufs_msgs__action__CheckForObjects_Goal__Sequence *
eufs_msgs__action__CheckForObjects_Goal__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_Goal__Sequence * array = (eufs_msgs__action__CheckForObjects_Goal__Sequence *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_Goal__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__action__CheckForObjects_Goal__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__action__CheckForObjects_Goal__Sequence__destroy(eufs_msgs__action__CheckForObjects_Goal__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__action__CheckForObjects_Goal__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__action__CheckForObjects_Goal__Sequence__are_equal(const eufs_msgs__action__CheckForObjects_Goal__Sequence * lhs, const eufs_msgs__action__CheckForObjects_Goal__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__action__CheckForObjects_Goal__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_Goal__Sequence__copy(
  const eufs_msgs__action__CheckForObjects_Goal__Sequence * input,
  eufs_msgs__action__CheckForObjects_Goal__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__action__CheckForObjects_Goal);
    eufs_msgs__action__CheckForObjects_Goal * data =
      (eufs_msgs__action__CheckForObjects_Goal *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__action__CheckForObjects_Goal__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__action__CheckForObjects_Goal__fini(&data[i]);
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
    if (!eufs_msgs__action__CheckForObjects_Goal__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `bounding_boxes`
#include "eufs_msgs/msg/detail/bounding_boxes__functions.h"

bool
eufs_msgs__action__CheckForObjects_Result__init(eufs_msgs__action__CheckForObjects_Result * msg)
{
  if (!msg) {
    return false;
  }
  // id
  // bounding_boxes
  if (!eufs_msgs__msg__BoundingBoxes__init(&msg->bounding_boxes)) {
    eufs_msgs__action__CheckForObjects_Result__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__action__CheckForObjects_Result__fini(eufs_msgs__action__CheckForObjects_Result * msg)
{
  if (!msg) {
    return;
  }
  // id
  // bounding_boxes
  eufs_msgs__msg__BoundingBoxes__fini(&msg->bounding_boxes);
}

bool
eufs_msgs__action__CheckForObjects_Result__are_equal(const eufs_msgs__action__CheckForObjects_Result * lhs, const eufs_msgs__action__CheckForObjects_Result * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // id
  if (lhs->id != rhs->id) {
    return false;
  }
  // bounding_boxes
  if (!eufs_msgs__msg__BoundingBoxes__are_equal(
      &(lhs->bounding_boxes), &(rhs->bounding_boxes)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_Result__copy(
  const eufs_msgs__action__CheckForObjects_Result * input,
  eufs_msgs__action__CheckForObjects_Result * output)
{
  if (!input || !output) {
    return false;
  }
  // id
  output->id = input->id;
  // bounding_boxes
  if (!eufs_msgs__msg__BoundingBoxes__copy(
      &(input->bounding_boxes), &(output->bounding_boxes)))
  {
    return false;
  }
  return true;
}

eufs_msgs__action__CheckForObjects_Result *
eufs_msgs__action__CheckForObjects_Result__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_Result * msg = (eufs_msgs__action__CheckForObjects_Result *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_Result), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__action__CheckForObjects_Result));
  bool success = eufs_msgs__action__CheckForObjects_Result__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__action__CheckForObjects_Result__destroy(eufs_msgs__action__CheckForObjects_Result * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__action__CheckForObjects_Result__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__action__CheckForObjects_Result__Sequence__init(eufs_msgs__action__CheckForObjects_Result__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_Result * data = NULL;

  if (size) {
    data = (eufs_msgs__action__CheckForObjects_Result *)allocator.zero_allocate(size, sizeof(eufs_msgs__action__CheckForObjects_Result), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__action__CheckForObjects_Result__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__action__CheckForObjects_Result__fini(&data[i - 1]);
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
eufs_msgs__action__CheckForObjects_Result__Sequence__fini(eufs_msgs__action__CheckForObjects_Result__Sequence * array)
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
      eufs_msgs__action__CheckForObjects_Result__fini(&array->data[i]);
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

eufs_msgs__action__CheckForObjects_Result__Sequence *
eufs_msgs__action__CheckForObjects_Result__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_Result__Sequence * array = (eufs_msgs__action__CheckForObjects_Result__Sequence *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_Result__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__action__CheckForObjects_Result__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__action__CheckForObjects_Result__Sequence__destroy(eufs_msgs__action__CheckForObjects_Result__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__action__CheckForObjects_Result__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__action__CheckForObjects_Result__Sequence__are_equal(const eufs_msgs__action__CheckForObjects_Result__Sequence * lhs, const eufs_msgs__action__CheckForObjects_Result__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__action__CheckForObjects_Result__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_Result__Sequence__copy(
  const eufs_msgs__action__CheckForObjects_Result__Sequence * input,
  eufs_msgs__action__CheckForObjects_Result__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__action__CheckForObjects_Result);
    eufs_msgs__action__CheckForObjects_Result * data =
      (eufs_msgs__action__CheckForObjects_Result *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__action__CheckForObjects_Result__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__action__CheckForObjects_Result__fini(&data[i]);
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
    if (!eufs_msgs__action__CheckForObjects_Result__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


bool
eufs_msgs__action__CheckForObjects_Feedback__init(eufs_msgs__action__CheckForObjects_Feedback * msg)
{
  if (!msg) {
    return false;
  }
  // structure_needs_at_least_one_member
  return true;
}

void
eufs_msgs__action__CheckForObjects_Feedback__fini(eufs_msgs__action__CheckForObjects_Feedback * msg)
{
  if (!msg) {
    return;
  }
  // structure_needs_at_least_one_member
}

bool
eufs_msgs__action__CheckForObjects_Feedback__are_equal(const eufs_msgs__action__CheckForObjects_Feedback * lhs, const eufs_msgs__action__CheckForObjects_Feedback * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // structure_needs_at_least_one_member
  if (lhs->structure_needs_at_least_one_member != rhs->structure_needs_at_least_one_member) {
    return false;
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_Feedback__copy(
  const eufs_msgs__action__CheckForObjects_Feedback * input,
  eufs_msgs__action__CheckForObjects_Feedback * output)
{
  if (!input || !output) {
    return false;
  }
  // structure_needs_at_least_one_member
  output->structure_needs_at_least_one_member = input->structure_needs_at_least_one_member;
  return true;
}

eufs_msgs__action__CheckForObjects_Feedback *
eufs_msgs__action__CheckForObjects_Feedback__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_Feedback * msg = (eufs_msgs__action__CheckForObjects_Feedback *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_Feedback), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__action__CheckForObjects_Feedback));
  bool success = eufs_msgs__action__CheckForObjects_Feedback__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__action__CheckForObjects_Feedback__destroy(eufs_msgs__action__CheckForObjects_Feedback * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__action__CheckForObjects_Feedback__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__action__CheckForObjects_Feedback__Sequence__init(eufs_msgs__action__CheckForObjects_Feedback__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_Feedback * data = NULL;

  if (size) {
    data = (eufs_msgs__action__CheckForObjects_Feedback *)allocator.zero_allocate(size, sizeof(eufs_msgs__action__CheckForObjects_Feedback), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__action__CheckForObjects_Feedback__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__action__CheckForObjects_Feedback__fini(&data[i - 1]);
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
eufs_msgs__action__CheckForObjects_Feedback__Sequence__fini(eufs_msgs__action__CheckForObjects_Feedback__Sequence * array)
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
      eufs_msgs__action__CheckForObjects_Feedback__fini(&array->data[i]);
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

eufs_msgs__action__CheckForObjects_Feedback__Sequence *
eufs_msgs__action__CheckForObjects_Feedback__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_Feedback__Sequence * array = (eufs_msgs__action__CheckForObjects_Feedback__Sequence *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_Feedback__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__action__CheckForObjects_Feedback__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__action__CheckForObjects_Feedback__Sequence__destroy(eufs_msgs__action__CheckForObjects_Feedback__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__action__CheckForObjects_Feedback__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__action__CheckForObjects_Feedback__Sequence__are_equal(const eufs_msgs__action__CheckForObjects_Feedback__Sequence * lhs, const eufs_msgs__action__CheckForObjects_Feedback__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__action__CheckForObjects_Feedback__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_Feedback__Sequence__copy(
  const eufs_msgs__action__CheckForObjects_Feedback__Sequence * input,
  eufs_msgs__action__CheckForObjects_Feedback__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__action__CheckForObjects_Feedback);
    eufs_msgs__action__CheckForObjects_Feedback * data =
      (eufs_msgs__action__CheckForObjects_Feedback *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__action__CheckForObjects_Feedback__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__action__CheckForObjects_Feedback__fini(&data[i]);
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
    if (!eufs_msgs__action__CheckForObjects_Feedback__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
#include "unique_identifier_msgs/msg/detail/uuid__functions.h"
// Member `goal`
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__functions.h"

bool
eufs_msgs__action__CheckForObjects_SendGoal_Request__init(eufs_msgs__action__CheckForObjects_SendGoal_Request * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    eufs_msgs__action__CheckForObjects_SendGoal_Request__fini(msg);
    return false;
  }
  // goal
  if (!eufs_msgs__action__CheckForObjects_Goal__init(&msg->goal)) {
    eufs_msgs__action__CheckForObjects_SendGoal_Request__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__action__CheckForObjects_SendGoal_Request__fini(eufs_msgs__action__CheckForObjects_SendGoal_Request * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
  // goal
  eufs_msgs__action__CheckForObjects_Goal__fini(&msg->goal);
}

bool
eufs_msgs__action__CheckForObjects_SendGoal_Request__are_equal(const eufs_msgs__action__CheckForObjects_SendGoal_Request * lhs, const eufs_msgs__action__CheckForObjects_SendGoal_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  // goal
  if (!eufs_msgs__action__CheckForObjects_Goal__are_equal(
      &(lhs->goal), &(rhs->goal)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_SendGoal_Request__copy(
  const eufs_msgs__action__CheckForObjects_SendGoal_Request * input,
  eufs_msgs__action__CheckForObjects_SendGoal_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  // goal
  if (!eufs_msgs__action__CheckForObjects_Goal__copy(
      &(input->goal), &(output->goal)))
  {
    return false;
  }
  return true;
}

eufs_msgs__action__CheckForObjects_SendGoal_Request *
eufs_msgs__action__CheckForObjects_SendGoal_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_SendGoal_Request * msg = (eufs_msgs__action__CheckForObjects_SendGoal_Request *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Request));
  bool success = eufs_msgs__action__CheckForObjects_SendGoal_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__action__CheckForObjects_SendGoal_Request__destroy(eufs_msgs__action__CheckForObjects_SendGoal_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__action__CheckForObjects_SendGoal_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence__init(eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_SendGoal_Request * data = NULL;

  if (size) {
    data = (eufs_msgs__action__CheckForObjects_SendGoal_Request *)allocator.zero_allocate(size, sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__action__CheckForObjects_SendGoal_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__action__CheckForObjects_SendGoal_Request__fini(&data[i - 1]);
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
eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence__fini(eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence * array)
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
      eufs_msgs__action__CheckForObjects_SendGoal_Request__fini(&array->data[i]);
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

eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence *
eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence * array = (eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence__destroy(eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence__are_equal(const eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence * lhs, const eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__action__CheckForObjects_SendGoal_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence__copy(
  const eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence * input,
  eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Request);
    eufs_msgs__action__CheckForObjects_SendGoal_Request * data =
      (eufs_msgs__action__CheckForObjects_SendGoal_Request *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__action__CheckForObjects_SendGoal_Request__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__action__CheckForObjects_SendGoal_Request__fini(&data[i]);
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
    if (!eufs_msgs__action__CheckForObjects_SendGoal_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `stamp`
#include "builtin_interfaces/msg/detail/time__functions.h"

bool
eufs_msgs__action__CheckForObjects_SendGoal_Response__init(eufs_msgs__action__CheckForObjects_SendGoal_Response * msg)
{
  if (!msg) {
    return false;
  }
  // accepted
  // stamp
  if (!builtin_interfaces__msg__Time__init(&msg->stamp)) {
    eufs_msgs__action__CheckForObjects_SendGoal_Response__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__action__CheckForObjects_SendGoal_Response__fini(eufs_msgs__action__CheckForObjects_SendGoal_Response * msg)
{
  if (!msg) {
    return;
  }
  // accepted
  // stamp
  builtin_interfaces__msg__Time__fini(&msg->stamp);
}

bool
eufs_msgs__action__CheckForObjects_SendGoal_Response__are_equal(const eufs_msgs__action__CheckForObjects_SendGoal_Response * lhs, const eufs_msgs__action__CheckForObjects_SendGoal_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // accepted
  if (lhs->accepted != rhs->accepted) {
    return false;
  }
  // stamp
  if (!builtin_interfaces__msg__Time__are_equal(
      &(lhs->stamp), &(rhs->stamp)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_SendGoal_Response__copy(
  const eufs_msgs__action__CheckForObjects_SendGoal_Response * input,
  eufs_msgs__action__CheckForObjects_SendGoal_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // accepted
  output->accepted = input->accepted;
  // stamp
  if (!builtin_interfaces__msg__Time__copy(
      &(input->stamp), &(output->stamp)))
  {
    return false;
  }
  return true;
}

eufs_msgs__action__CheckForObjects_SendGoal_Response *
eufs_msgs__action__CheckForObjects_SendGoal_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_SendGoal_Response * msg = (eufs_msgs__action__CheckForObjects_SendGoal_Response *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Response));
  bool success = eufs_msgs__action__CheckForObjects_SendGoal_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__action__CheckForObjects_SendGoal_Response__destroy(eufs_msgs__action__CheckForObjects_SendGoal_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__action__CheckForObjects_SendGoal_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence__init(eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_SendGoal_Response * data = NULL;

  if (size) {
    data = (eufs_msgs__action__CheckForObjects_SendGoal_Response *)allocator.zero_allocate(size, sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__action__CheckForObjects_SendGoal_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__action__CheckForObjects_SendGoal_Response__fini(&data[i - 1]);
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
eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence__fini(eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence * array)
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
      eufs_msgs__action__CheckForObjects_SendGoal_Response__fini(&array->data[i]);
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

eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence *
eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence * array = (eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence__destroy(eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence__are_equal(const eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence * lhs, const eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__action__CheckForObjects_SendGoal_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence__copy(
  const eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence * input,
  eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__action__CheckForObjects_SendGoal_Response);
    eufs_msgs__action__CheckForObjects_SendGoal_Response * data =
      (eufs_msgs__action__CheckForObjects_SendGoal_Response *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__action__CheckForObjects_SendGoal_Response__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__action__CheckForObjects_SendGoal_Response__fini(&data[i]);
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
    if (!eufs_msgs__action__CheckForObjects_SendGoal_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__functions.h"

bool
eufs_msgs__action__CheckForObjects_GetResult_Request__init(eufs_msgs__action__CheckForObjects_GetResult_Request * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    eufs_msgs__action__CheckForObjects_GetResult_Request__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__action__CheckForObjects_GetResult_Request__fini(eufs_msgs__action__CheckForObjects_GetResult_Request * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
}

bool
eufs_msgs__action__CheckForObjects_GetResult_Request__are_equal(const eufs_msgs__action__CheckForObjects_GetResult_Request * lhs, const eufs_msgs__action__CheckForObjects_GetResult_Request * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_GetResult_Request__copy(
  const eufs_msgs__action__CheckForObjects_GetResult_Request * input,
  eufs_msgs__action__CheckForObjects_GetResult_Request * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  return true;
}

eufs_msgs__action__CheckForObjects_GetResult_Request *
eufs_msgs__action__CheckForObjects_GetResult_Request__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_GetResult_Request * msg = (eufs_msgs__action__CheckForObjects_GetResult_Request *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_GetResult_Request), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__action__CheckForObjects_GetResult_Request));
  bool success = eufs_msgs__action__CheckForObjects_GetResult_Request__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__action__CheckForObjects_GetResult_Request__destroy(eufs_msgs__action__CheckForObjects_GetResult_Request * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__action__CheckForObjects_GetResult_Request__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence__init(eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_GetResult_Request * data = NULL;

  if (size) {
    data = (eufs_msgs__action__CheckForObjects_GetResult_Request *)allocator.zero_allocate(size, sizeof(eufs_msgs__action__CheckForObjects_GetResult_Request), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__action__CheckForObjects_GetResult_Request__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__action__CheckForObjects_GetResult_Request__fini(&data[i - 1]);
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
eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence__fini(eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence * array)
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
      eufs_msgs__action__CheckForObjects_GetResult_Request__fini(&array->data[i]);
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

eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence *
eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence * array = (eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence__destroy(eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence__are_equal(const eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence * lhs, const eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__action__CheckForObjects_GetResult_Request__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence__copy(
  const eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence * input,
  eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__action__CheckForObjects_GetResult_Request);
    eufs_msgs__action__CheckForObjects_GetResult_Request * data =
      (eufs_msgs__action__CheckForObjects_GetResult_Request *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__action__CheckForObjects_GetResult_Request__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__action__CheckForObjects_GetResult_Request__fini(&data[i]);
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
    if (!eufs_msgs__action__CheckForObjects_GetResult_Request__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `result`
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__functions.h"

bool
eufs_msgs__action__CheckForObjects_GetResult_Response__init(eufs_msgs__action__CheckForObjects_GetResult_Response * msg)
{
  if (!msg) {
    return false;
  }
  // status
  // result
  if (!eufs_msgs__action__CheckForObjects_Result__init(&msg->result)) {
    eufs_msgs__action__CheckForObjects_GetResult_Response__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__action__CheckForObjects_GetResult_Response__fini(eufs_msgs__action__CheckForObjects_GetResult_Response * msg)
{
  if (!msg) {
    return;
  }
  // status
  // result
  eufs_msgs__action__CheckForObjects_Result__fini(&msg->result);
}

bool
eufs_msgs__action__CheckForObjects_GetResult_Response__are_equal(const eufs_msgs__action__CheckForObjects_GetResult_Response * lhs, const eufs_msgs__action__CheckForObjects_GetResult_Response * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // status
  if (lhs->status != rhs->status) {
    return false;
  }
  // result
  if (!eufs_msgs__action__CheckForObjects_Result__are_equal(
      &(lhs->result), &(rhs->result)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_GetResult_Response__copy(
  const eufs_msgs__action__CheckForObjects_GetResult_Response * input,
  eufs_msgs__action__CheckForObjects_GetResult_Response * output)
{
  if (!input || !output) {
    return false;
  }
  // status
  output->status = input->status;
  // result
  if (!eufs_msgs__action__CheckForObjects_Result__copy(
      &(input->result), &(output->result)))
  {
    return false;
  }
  return true;
}

eufs_msgs__action__CheckForObjects_GetResult_Response *
eufs_msgs__action__CheckForObjects_GetResult_Response__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_GetResult_Response * msg = (eufs_msgs__action__CheckForObjects_GetResult_Response *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_GetResult_Response), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__action__CheckForObjects_GetResult_Response));
  bool success = eufs_msgs__action__CheckForObjects_GetResult_Response__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__action__CheckForObjects_GetResult_Response__destroy(eufs_msgs__action__CheckForObjects_GetResult_Response * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__action__CheckForObjects_GetResult_Response__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence__init(eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_GetResult_Response * data = NULL;

  if (size) {
    data = (eufs_msgs__action__CheckForObjects_GetResult_Response *)allocator.zero_allocate(size, sizeof(eufs_msgs__action__CheckForObjects_GetResult_Response), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__action__CheckForObjects_GetResult_Response__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__action__CheckForObjects_GetResult_Response__fini(&data[i - 1]);
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
eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence__fini(eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence * array)
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
      eufs_msgs__action__CheckForObjects_GetResult_Response__fini(&array->data[i]);
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

eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence *
eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence * array = (eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence__destroy(eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence__are_equal(const eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence * lhs, const eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__action__CheckForObjects_GetResult_Response__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence__copy(
  const eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence * input,
  eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__action__CheckForObjects_GetResult_Response);
    eufs_msgs__action__CheckForObjects_GetResult_Response * data =
      (eufs_msgs__action__CheckForObjects_GetResult_Response *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__action__CheckForObjects_GetResult_Response__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__action__CheckForObjects_GetResult_Response__fini(&data[i]);
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
    if (!eufs_msgs__action__CheckForObjects_GetResult_Response__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}


// Include directives for member types
// Member `goal_id`
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__functions.h"
// Member `feedback`
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__functions.h"

bool
eufs_msgs__action__CheckForObjects_FeedbackMessage__init(eufs_msgs__action__CheckForObjects_FeedbackMessage * msg)
{
  if (!msg) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__init(&msg->goal_id)) {
    eufs_msgs__action__CheckForObjects_FeedbackMessage__fini(msg);
    return false;
  }
  // feedback
  if (!eufs_msgs__action__CheckForObjects_Feedback__init(&msg->feedback)) {
    eufs_msgs__action__CheckForObjects_FeedbackMessage__fini(msg);
    return false;
  }
  return true;
}

void
eufs_msgs__action__CheckForObjects_FeedbackMessage__fini(eufs_msgs__action__CheckForObjects_FeedbackMessage * msg)
{
  if (!msg) {
    return;
  }
  // goal_id
  unique_identifier_msgs__msg__UUID__fini(&msg->goal_id);
  // feedback
  eufs_msgs__action__CheckForObjects_Feedback__fini(&msg->feedback);
}

bool
eufs_msgs__action__CheckForObjects_FeedbackMessage__are_equal(const eufs_msgs__action__CheckForObjects_FeedbackMessage * lhs, const eufs_msgs__action__CheckForObjects_FeedbackMessage * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__are_equal(
      &(lhs->goal_id), &(rhs->goal_id)))
  {
    return false;
  }
  // feedback
  if (!eufs_msgs__action__CheckForObjects_Feedback__are_equal(
      &(lhs->feedback), &(rhs->feedback)))
  {
    return false;
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_FeedbackMessage__copy(
  const eufs_msgs__action__CheckForObjects_FeedbackMessage * input,
  eufs_msgs__action__CheckForObjects_FeedbackMessage * output)
{
  if (!input || !output) {
    return false;
  }
  // goal_id
  if (!unique_identifier_msgs__msg__UUID__copy(
      &(input->goal_id), &(output->goal_id)))
  {
    return false;
  }
  // feedback
  if (!eufs_msgs__action__CheckForObjects_Feedback__copy(
      &(input->feedback), &(output->feedback)))
  {
    return false;
  }
  return true;
}

eufs_msgs__action__CheckForObjects_FeedbackMessage *
eufs_msgs__action__CheckForObjects_FeedbackMessage__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_FeedbackMessage * msg = (eufs_msgs__action__CheckForObjects_FeedbackMessage *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_FeedbackMessage), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(eufs_msgs__action__CheckForObjects_FeedbackMessage));
  bool success = eufs_msgs__action__CheckForObjects_FeedbackMessage__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
eufs_msgs__action__CheckForObjects_FeedbackMessage__destroy(eufs_msgs__action__CheckForObjects_FeedbackMessage * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    eufs_msgs__action__CheckForObjects_FeedbackMessage__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence__init(eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_FeedbackMessage * data = NULL;

  if (size) {
    data = (eufs_msgs__action__CheckForObjects_FeedbackMessage *)allocator.zero_allocate(size, sizeof(eufs_msgs__action__CheckForObjects_FeedbackMessage), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = eufs_msgs__action__CheckForObjects_FeedbackMessage__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        eufs_msgs__action__CheckForObjects_FeedbackMessage__fini(&data[i - 1]);
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
eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence__fini(eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence * array)
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
      eufs_msgs__action__CheckForObjects_FeedbackMessage__fini(&array->data[i]);
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

eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence *
eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence * array = (eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence *)allocator.allocate(sizeof(eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence__destroy(eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence__are_equal(const eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence * lhs, const eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!eufs_msgs__action__CheckForObjects_FeedbackMessage__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence__copy(
  const eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence * input,
  eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(eufs_msgs__action__CheckForObjects_FeedbackMessage);
    eufs_msgs__action__CheckForObjects_FeedbackMessage * data =
      (eufs_msgs__action__CheckForObjects_FeedbackMessage *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!eufs_msgs__action__CheckForObjects_FeedbackMessage__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          eufs_msgs__action__CheckForObjects_FeedbackMessage__fini(&data[i]);
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
    if (!eufs_msgs__action__CheckForObjects_FeedbackMessage__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
