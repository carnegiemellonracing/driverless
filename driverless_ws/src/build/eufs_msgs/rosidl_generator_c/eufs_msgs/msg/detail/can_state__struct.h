// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:msg/CanState.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__CAN_STATE__STRUCT_H_
#define EUFS_MSGS__MSG__DETAIL__CAN_STATE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

/// Constant 'AS_OFF'.
enum
{
  eufs_msgs__msg__CanState__AS_OFF = 0
};

/// Constant 'AS_READY'.
enum
{
  eufs_msgs__msg__CanState__AS_READY = 1
};

/// Constant 'AS_DRIVING'.
enum
{
  eufs_msgs__msg__CanState__AS_DRIVING = 2
};

/// Constant 'AS_EMERGENCY_BRAKE'.
enum
{
  eufs_msgs__msg__CanState__AS_EMERGENCY_BRAKE = 3
};

/// Constant 'AS_FINISHED'.
enum
{
  eufs_msgs__msg__CanState__AS_FINISHED = 4
};

/// Constant 'AMI_NOT_SELECTED'.
enum
{
  eufs_msgs__msg__CanState__AMI_NOT_SELECTED = 10
};

/// Constant 'AMI_ACCELERATION'.
enum
{
  eufs_msgs__msg__CanState__AMI_ACCELERATION = 11
};

/// Constant 'AMI_SKIDPAD'.
enum
{
  eufs_msgs__msg__CanState__AMI_SKIDPAD = 12
};

/// Constant 'AMI_AUTOCROSS'.
enum
{
  eufs_msgs__msg__CanState__AMI_AUTOCROSS = 13
};

/// Constant 'AMI_TRACK_DRIVE'.
enum
{
  eufs_msgs__msg__CanState__AMI_TRACK_DRIVE = 14
};

/// Constant 'AMI_AUTONOMOUS_DEMO'.
enum
{
  eufs_msgs__msg__CanState__AMI_AUTONOMOUS_DEMO = 15
};

/// Constant 'AMI_ADS_INSPECTION'.
enum
{
  eufs_msgs__msg__CanState__AMI_ADS_INSPECTION = 16
};

/// Constant 'AMI_ADS_EBS'.
enum
{
  eufs_msgs__msg__CanState__AMI_ADS_EBS = 17
};

/// Constant 'AMI_DDT_INSPECTION_A'.
enum
{
  eufs_msgs__msg__CanState__AMI_DDT_INSPECTION_A = 18
};

/// Constant 'AMI_DDT_INSPECTION_B'.
enum
{
  eufs_msgs__msg__CanState__AMI_DDT_INSPECTION_B = 19
};

/// Constant 'AMI_JOYSTICK'.
enum
{
  eufs_msgs__msg__CanState__AMI_JOYSTICK = 20
};

/// Constant 'AMI_MANUAL'.
enum
{
  eufs_msgs__msg__CanState__AMI_MANUAL = 21
};

// Struct defined in msg/CanState in the package eufs_msgs.
typedef struct eufs_msgs__msg__CanState
{
  uint16_t as_state;
  uint16_t ami_state;
} eufs_msgs__msg__CanState;

// Struct for a sequence of eufs_msgs__msg__CanState.
typedef struct eufs_msgs__msg__CanState__Sequence
{
  eufs_msgs__msg__CanState * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__msg__CanState__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__MSG__DETAIL__CAN_STATE__STRUCT_H_
