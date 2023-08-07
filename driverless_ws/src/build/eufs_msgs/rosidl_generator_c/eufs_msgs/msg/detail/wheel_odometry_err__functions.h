// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from eufs_msgs:msg/WheelOdometryErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__WHEEL_ODOMETRY_ERR__FUNCTIONS_H_
#define EUFS_MSGS__MSG__DETAIL__WHEEL_ODOMETRY_ERR__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "eufs_msgs/msg/rosidl_generator_c__visibility_control.h"

#include "eufs_msgs/msg/detail/wheel_odometry_err__struct.h"

/// Initialize msg/WheelOdometryErr message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * eufs_msgs__msg__WheelOdometryErr
 * )) before or use
 * eufs_msgs__msg__WheelOdometryErr__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__WheelOdometryErr__init(eufs_msgs__msg__WheelOdometryErr * msg);

/// Finalize msg/WheelOdometryErr message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__WheelOdometryErr__fini(eufs_msgs__msg__WheelOdometryErr * msg);

/// Create msg/WheelOdometryErr message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * eufs_msgs__msg__WheelOdometryErr__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
eufs_msgs__msg__WheelOdometryErr *
eufs_msgs__msg__WheelOdometryErr__create();

/// Destroy msg/WheelOdometryErr message.
/**
 * It calls
 * eufs_msgs__msg__WheelOdometryErr__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__WheelOdometryErr__destroy(eufs_msgs__msg__WheelOdometryErr * msg);

/// Check for msg/WheelOdometryErr message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__WheelOdometryErr__are_equal(const eufs_msgs__msg__WheelOdometryErr * lhs, const eufs_msgs__msg__WheelOdometryErr * rhs);

/// Copy a msg/WheelOdometryErr message.
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
eufs_msgs__msg__WheelOdometryErr__copy(
  const eufs_msgs__msg__WheelOdometryErr * input,
  eufs_msgs__msg__WheelOdometryErr * output);

/// Initialize array of msg/WheelOdometryErr messages.
/**
 * It allocates the memory for the number of elements and calls
 * eufs_msgs__msg__WheelOdometryErr__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__WheelOdometryErr__Sequence__init(eufs_msgs__msg__WheelOdometryErr__Sequence * array, size_t size);

/// Finalize array of msg/WheelOdometryErr messages.
/**
 * It calls
 * eufs_msgs__msg__WheelOdometryErr__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__WheelOdometryErr__Sequence__fini(eufs_msgs__msg__WheelOdometryErr__Sequence * array);

/// Create array of msg/WheelOdometryErr messages.
/**
 * It allocates the memory for the array and calls
 * eufs_msgs__msg__WheelOdometryErr__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
eufs_msgs__msg__WheelOdometryErr__Sequence *
eufs_msgs__msg__WheelOdometryErr__Sequence__create(size_t size);

/// Destroy array of msg/WheelOdometryErr messages.
/**
 * It calls
 * eufs_msgs__msg__WheelOdometryErr__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
void
eufs_msgs__msg__WheelOdometryErr__Sequence__destroy(eufs_msgs__msg__WheelOdometryErr__Sequence * array);

/// Check for msg/WheelOdometryErr message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_eufs_msgs
bool
eufs_msgs__msg__WheelOdometryErr__Sequence__are_equal(const eufs_msgs__msg__WheelOdometryErr__Sequence * lhs, const eufs_msgs__msg__WheelOdometryErr__Sequence * rhs);

/// Copy an array of msg/WheelOdometryErr messages.
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
eufs_msgs__msg__WheelOdometryErr__Sequence__copy(
  const eufs_msgs__msg__WheelOdometryErr__Sequence * input,
  eufs_msgs__msg__WheelOdometryErr__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__WHEEL_ODOMETRY_ERR__FUNCTIONS_H_
