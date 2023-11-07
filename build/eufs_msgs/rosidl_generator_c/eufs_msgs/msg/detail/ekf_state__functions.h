// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from eufs_msgs:msg/EKFState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__EKF_STATE__FUNCTIONS_H_
#define EUFS_MSGS__MSG__DETAIL__EKF_STATE__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "eufs_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "eufs_msgs/msg/detail/ekf_state__struct.h"

/// Initialize msg/EKFState message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * eufs_msgs__msg__EKFState
 * )) before or use
 * eufs_msgs__msg__EKFState__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__EKFState__init(eufs_msgs__msg__EKFState * msg);

/// Finalize msg/EKFState message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__EKFState__fini(eufs_msgs__msg__EKFState * msg);

/// Create msg/EKFState message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * eufs_msgs__msg__EKFState__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
eufs_msgs__msg__EKFState *
eufs_msgs__msg__EKFState__create();

/// Destroy msg/EKFState message.
/**
 * It calls
 * eufs_msgs__msg__EKFState__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__EKFState__destroy(eufs_msgs__msg__EKFState * msg);

/// Check for msg/EKFState message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__EKFState__are_equal(const eufs_msgs__msg__EKFState * lhs, const eufs_msgs__msg__EKFState * rhs);

/// Copy a msg/EKFState message.
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
eufs_msgs__msg__EKFState__copy(
  const eufs_msgs__msg__EKFState * input,
  eufs_msgs__msg__EKFState * output);

/// Initialize array of msg/EKFState messages.
/**
 * It allocates the memory for the number of elements and calls
 * eufs_msgs__msg__EKFState__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__EKFState__Sequence__init(eufs_msgs__msg__EKFState__Sequence * array, size_t size);

/// Finalize array of msg/EKFState messages.
/**
 * It calls
 * eufs_msgs__msg__EKFState__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__EKFState__Sequence__fini(eufs_msgs__msg__EKFState__Sequence * array);

/// Create array of msg/EKFState messages.
/**
 * It allocates the memory for the array and calls
 * eufs_msgs__msg__EKFState__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
eufs_msgs__msg__EKFState__Sequence *
eufs_msgs__msg__EKFState__Sequence__create(size_t size);

/// Destroy array of msg/EKFState messages.
/**
 * It calls
 * eufs_msgs__msg__EKFState__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__EKFState__Sequence__destroy(eufs_msgs__msg__EKFState__Sequence * array);

/// Check for msg/EKFState message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__EKFState__Sequence__are_equal(const eufs_msgs__msg__EKFState__Sequence * lhs, const eufs_msgs__msg__EKFState__Sequence * rhs);

/// Copy an array of msg/EKFState messages.
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
eufs_msgs__msg__EKFState__Sequence__copy(
  const eufs_msgs__msg__EKFState__Sequence * input,
  eufs_msgs__msg__EKFState__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__EKF_STATE__FUNCTIONS_H_
