// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from eufs_msgs:msg/Waypoint.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WAYPOINT__FUNCTIONS_H_
#define EUFS_MSGS__MSG__DETAIL__WAYPOINT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "eufs_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "eufs_msgs/msg/detail/waypoint__struct.h"

/// Initialize msg/Waypoint message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * eufs_msgs__msg__Waypoint
 * )) before or use
 * eufs_msgs__msg__Waypoint__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__Waypoint__init(eufs_msgs__msg__Waypoint * msg);

/// Finalize msg/Waypoint message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__Waypoint__fini(eufs_msgs__msg__Waypoint * msg);

/// Create msg/Waypoint message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * eufs_msgs__msg__Waypoint__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
eufs_msgs__msg__Waypoint *
eufs_msgs__msg__Waypoint__create();

/// Destroy msg/Waypoint message.
/**
 * It calls
 * eufs_msgs__msg__Waypoint__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__Waypoint__destroy(eufs_msgs__msg__Waypoint * msg);

/// Check for msg/Waypoint message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__Waypoint__are_equal(const eufs_msgs__msg__Waypoint * lhs, const eufs_msgs__msg__Waypoint * rhs);

/// Copy a msg/Waypoint message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__Waypoint__copy(
  const eufs_msgs__msg__Waypoint * input,
  eufs_msgs__msg__Waypoint * output);

/// Initialize array of msg/Waypoint messages.
/**
 * It allocates the memory for the number of elements and calls
 * eufs_msgs__msg__Waypoint__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__Waypoint__Sequence__init(eufs_msgs__msg__Waypoint__Sequence * array, size_t size);

/// Finalize array of msg/Waypoint messages.
/**
 * It calls
 * eufs_msgs__msg__Waypoint__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__Waypoint__Sequence__fini(eufs_msgs__msg__Waypoint__Sequence * array);

/// Create array of msg/Waypoint messages.
/**
 * It allocates the memory for the array and calls
 * eufs_msgs__msg__Waypoint__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
eufs_msgs__msg__Waypoint__Sequence *
eufs_msgs__msg__Waypoint__Sequence__create(size_t size);

/// Destroy array of msg/Waypoint messages.
/**
 * It calls
 * eufs_msgs__msg__Waypoint__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__Waypoint__Sequence__destroy(eufs_msgs__msg__Waypoint__Sequence * array);

/// Check for msg/Waypoint message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__Waypoint__Sequence__are_equal(const eufs_msgs__msg__Waypoint__Sequence * lhs, const eufs_msgs__msg__Waypoint__Sequence * rhs);

/// Copy an array of msg/Waypoint messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__Waypoint__Sequence__copy(
  const eufs_msgs__msg__Waypoint__Sequence * input,
  eufs_msgs__msg__Waypoint__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__WAYPOINT__FUNCTIONS_H_
