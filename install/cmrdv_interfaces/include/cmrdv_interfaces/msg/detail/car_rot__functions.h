// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from cmrdv_interfaces:msg/CarROT.idl
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__FUNCTIONS_H_
#define CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "cmrdv_interfaces/msg/rosidl_generator_c__visibility_control.h"

#include "cmrdv_interfaces/msg/detail/car_rot__struct.h"

/// Initialize msg/CarROT message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * cmrdv_interfaces__msg__CarROT
 * )) before or use
 * cmrdv_interfaces__msg__CarROT__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__CarROT__init(cmrdv_interfaces__msg__CarROT * msg);

/// Finalize msg/CarROT message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__CarROT__fini(cmrdv_interfaces__msg__CarROT * msg);

/// Create msg/CarROT message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * cmrdv_interfaces__msg__CarROT__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
cmrdv_interfaces__msg__CarROT *
cmrdv_interfaces__msg__CarROT__create();

/// Destroy msg/CarROT message.
/**
 * It calls
 * cmrdv_interfaces__msg__CarROT__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__CarROT__destroy(cmrdv_interfaces__msg__CarROT * msg);

/// Check for msg/CarROT message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__CarROT__are_equal(const cmrdv_interfaces__msg__CarROT * lhs, const cmrdv_interfaces__msg__CarROT * rhs);

/// Copy a msg/CarROT message.
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
cmrdv_interfaces__msg__CarROT__copy(
  const cmrdv_interfaces__msg__CarROT * input,
  cmrdv_interfaces__msg__CarROT * output);

/// Initialize array of msg/CarROT messages.
/**
 * It allocates the memory for the number of elements and calls
 * cmrdv_interfaces__msg__CarROT__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__CarROT__Sequence__init(cmrdv_interfaces__msg__CarROT__Sequence * array, size_t size);

/// Finalize array of msg/CarROT messages.
/**
 * It calls
 * cmrdv_interfaces__msg__CarROT__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__CarROT__Sequence__fini(cmrdv_interfaces__msg__CarROT__Sequence * array);

/// Create array of msg/CarROT messages.
/**
 * It allocates the memory for the array and calls
 * cmrdv_interfaces__msg__CarROT__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
cmrdv_interfaces__msg__CarROT__Sequence *
cmrdv_interfaces__msg__CarROT__Sequence__create(size_t size);

/// Destroy array of msg/CarROT messages.
/**
 * It calls
 * cmrdv_interfaces__msg__CarROT__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
void
cmrdv_interfaces__msg__CarROT__Sequence__destroy(cmrdv_interfaces__msg__CarROT__Sequence * array);

/// Check for msg/CarROT message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_cmrdv_interfaces
bool
cmrdv_interfaces__msg__CarROT__Sequence__are_equal(const cmrdv_interfaces__msg__CarROT__Sequence * lhs, const cmrdv_interfaces__msg__CarROT__Sequence * rhs);

/// Copy an array of msg/CarROT messages.
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
cmrdv_interfaces__msg__CarROT__Sequence__copy(
  const cmrdv_interfaces__msg__CarROT__Sequence * input,
  cmrdv_interfaces__msg__CarROT__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__DETAIL__CAR_ROT__FUNCTIONS_H_