// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from cmrdv_interfaces:msg/DataFrame.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__FUNCTIONS_H_
#define CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "cmrdv_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "cmrdv_interfaces/msg/detail/data_frame__struct.h"

/// Initialize msg/DataFrame message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * cmrdv_interfaces__msg__DataFrame
 * )) before or use
 * cmrdv_interfaces__msg__DataFrame__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__DataFrame__init(cmrdv_interfaces__msg__DataFrame * msg);

/// Finalize msg/DataFrame message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__DataFrame__fini(cmrdv_interfaces__msg__DataFrame * msg);

/// Create msg/DataFrame message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * cmrdv_interfaces__msg__DataFrame__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
cmrdv_interfaces__msg__DataFrame *
cmrdv_interfaces__msg__DataFrame__create();

/// Destroy msg/DataFrame message.
/**
 * It calls
 * cmrdv_interfaces__msg__DataFrame__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__DataFrame__destroy(cmrdv_interfaces__msg__DataFrame * msg);

/// Check for msg/DataFrame message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__DataFrame__are_equal(const cmrdv_interfaces__msg__DataFrame * lhs, const cmrdv_interfaces__msg__DataFrame * rhs);

/// Copy a msg/DataFrame message.
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
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__DataFrame__copy(
  const cmrdv_interfaces__msg__DataFrame * input,
  cmrdv_interfaces__msg__DataFrame * output);

/// Initialize array of msg/DataFrame messages.
/**
 * It allocates the memory for the number of elements and calls
 * cmrdv_interfaces__msg__DataFrame__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__DataFrame__Sequence__init(cmrdv_interfaces__msg__DataFrame__Sequence * array, size_t size);

/// Finalize array of msg/DataFrame messages.
/**
 * It calls
 * cmrdv_interfaces__msg__DataFrame__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__DataFrame__Sequence__fini(cmrdv_interfaces__msg__DataFrame__Sequence * array);

/// Create array of msg/DataFrame messages.
/**
 * It allocates the memory for the array and calls
 * cmrdv_interfaces__msg__DataFrame__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
cmrdv_interfaces__msg__DataFrame__Sequence *
cmrdv_interfaces__msg__DataFrame__Sequence__create(size_t size);

/// Destroy array of msg/DataFrame messages.
/**
 * It calls
 * cmrdv_interfaces__msg__DataFrame__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__DataFrame__Sequence__destroy(cmrdv_interfaces__msg__DataFrame__Sequence * array);

/// Check for msg/DataFrame message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__DataFrame__Sequence__are_equal(const cmrdv_interfaces__msg__DataFrame__Sequence * lhs, const cmrdv_interfaces__msg__DataFrame__Sequence * rhs);

/// Copy an array of msg/DataFrame messages.
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
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__DataFrame__Sequence__copy(
  const cmrdv_interfaces__msg__DataFrame__Sequence * input,
  cmrdv_interfaces__msg__DataFrame__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__DATA_FRAME__FUNCTIONS_H_
