// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from cmrdv_interfaces:msg/ConePositions.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__FUNCTIONS_H_
#define CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "cmrdv_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "cmrdv_interfaces/msg/detail/cone_positions__struct.h"

/// Initialize msg/ConePositions message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * cmrdv_interfaces__msg__ConePositions
 * )) before or use
 * cmrdv_interfaces__msg__ConePositions__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__ConePositions__init(cmrdv_interfaces__msg__ConePositions * msg);

/// Finalize msg/ConePositions message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__ConePositions__fini(cmrdv_interfaces__msg__ConePositions * msg);

/// Create msg/ConePositions message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * cmrdv_interfaces__msg__ConePositions__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
cmrdv_interfaces__msg__ConePositions *
cmrdv_interfaces__msg__ConePositions__create();

/// Destroy msg/ConePositions message.
/**
 * It calls
 * cmrdv_interfaces__msg__ConePositions__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__ConePositions__destroy(cmrdv_interfaces__msg__ConePositions * msg);

/// Check for msg/ConePositions message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__ConePositions__are_equal(const cmrdv_interfaces__msg__ConePositions * lhs, const cmrdv_interfaces__msg__ConePositions * rhs);

/// Copy a msg/ConePositions message.
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
cmrdv_interfaces__msg__ConePositions__copy(
  const cmrdv_interfaces__msg__ConePositions * input,
  cmrdv_interfaces__msg__ConePositions * output);

/// Initialize array of msg/ConePositions messages.
/**
 * It allocates the memory for the number of elements and calls
 * cmrdv_interfaces__msg__ConePositions__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__ConePositions__Sequence__init(cmrdv_interfaces__msg__ConePositions__Sequence * array, size_t size);

/// Finalize array of msg/ConePositions messages.
/**
 * It calls
 * cmrdv_interfaces__msg__ConePositions__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__ConePositions__Sequence__fini(cmrdv_interfaces__msg__ConePositions__Sequence * array);

/// Create array of msg/ConePositions messages.
/**
 * It allocates the memory for the array and calls
 * cmrdv_interfaces__msg__ConePositions__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
cmrdv_interfaces__msg__ConePositions__Sequence *
cmrdv_interfaces__msg__ConePositions__Sequence__create(size_t size);

/// Destroy array of msg/ConePositions messages.
/**
 * It calls
 * cmrdv_interfaces__msg__ConePositions__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__ConePositions__Sequence__destroy(cmrdv_interfaces__msg__ConePositions__Sequence * array);

/// Check for msg/ConePositions message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__ConePositions__Sequence__are_equal(const cmrdv_interfaces__msg__ConePositions__Sequence * lhs, const cmrdv_interfaces__msg__ConePositions__Sequence * rhs);

/// Copy an array of msg/ConePositions messages.
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
cmrdv_interfaces__msg__ConePositions__Sequence__copy(
  const cmrdv_interfaces__msg__ConePositions__Sequence * input,
  cmrdv_interfaces__msg__ConePositions__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CONE_POSITIONS__FUNCTIONS_H_