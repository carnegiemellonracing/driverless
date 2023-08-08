// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from eufs_msgs:action/CheckForObjects.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__ACTION__DETAIL__CHECK_FOR_OBJECTS__STRUCT_H_
#define EUFS_MSGS__ACTION__DETAIL__CHECK_FOR_OBJECTS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'image'
#include "sensor_msgs/msg/detail/image__struct.h"

// Struct defined in action/CheckForObjects in the package eufs_msgs.
typedef struct eufs_msgs__action__CheckForObjects_Goal
{
  int16_t id;
  sensor_msgs__msg__Image image;
} eufs_msgs__action__CheckForObjects_Goal;

// Struct for a sequence of eufs_msgs__action__CheckForObjects_Goal.
typedef struct eufs_msgs__action__CheckForObjects_Goal__Sequence
{
  eufs_msgs__action__CheckForObjects_Goal * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__action__CheckForObjects_Goal__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'bounding_boxes'
#include "eufs_msgs/msg/detail/bounding_boxes__struct.h"

// Struct defined in action/CheckForObjects in the package eufs_msgs.
typedef struct eufs_msgs__action__CheckForObjects_Result
{
  int16_t id;
  eufs_msgs__msg__BoundingBoxes bounding_boxes;
} eufs_msgs__action__CheckForObjects_Result;

// Struct for a sequence of eufs_msgs__action__CheckForObjects_Result.
typedef struct eufs_msgs__action__CheckForObjects_Result__Sequence
{
  eufs_msgs__action__CheckForObjects_Result * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__action__CheckForObjects_Result__Sequence;


// Constants defined in the message

// Struct defined in action/CheckForObjects in the package eufs_msgs.
typedef struct eufs_msgs__action__CheckForObjects_Feedback
{
  uint8_t structure_needs_at_least_one_member;
} eufs_msgs__action__CheckForObjects_Feedback;

// Struct for a sequence of eufs_msgs__action__CheckForObjects_Feedback.
typedef struct eufs_msgs__action__CheckForObjects_Feedback__Sequence
{
  eufs_msgs__action__CheckForObjects_Feedback * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__action__CheckForObjects_Feedback__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
#include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'goal'
#include "eufs_msgs/action/detail/check_for_objects__struct.h"

// Struct defined in action/CheckForObjects in the package eufs_msgs.
typedef struct eufs_msgs__action__CheckForObjects_SendGoal_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
  eufs_msgs__action__CheckForObjects_Goal goal;
} eufs_msgs__action__CheckForObjects_SendGoal_Request;

// Struct for a sequence of eufs_msgs__action__CheckForObjects_SendGoal_Request.
typedef struct eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence
{
  eufs_msgs__action__CheckForObjects_SendGoal_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__action__CheckForObjects_SendGoal_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'stamp'
#include "builtin_interfaces/msg/detail/time__struct.h"

// Struct defined in action/CheckForObjects in the package eufs_msgs.
typedef struct eufs_msgs__action__CheckForObjects_SendGoal_Response
{
  bool accepted;
  builtin_interfaces__msg__Time stamp;
} eufs_msgs__action__CheckForObjects_SendGoal_Response;

// Struct for a sequence of eufs_msgs__action__CheckForObjects_SendGoal_Response.
typedef struct eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence
{
  eufs_msgs__action__CheckForObjects_SendGoal_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__action__CheckForObjects_SendGoal_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"

// Struct defined in action/CheckForObjects in the package eufs_msgs.
typedef struct eufs_msgs__action__CheckForObjects_GetResult_Request
{
  unique_identifier_msgs__msg__UUID goal_id;
} eufs_msgs__action__CheckForObjects_GetResult_Request;

// Struct for a sequence of eufs_msgs__action__CheckForObjects_GetResult_Request.
typedef struct eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence
{
  eufs_msgs__action__CheckForObjects_GetResult_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__action__CheckForObjects_GetResult_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'result'
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__struct.h"

// Struct defined in action/CheckForObjects in the package eufs_msgs.
typedef struct eufs_msgs__action__CheckForObjects_GetResult_Response
{
  int8_t status;
  eufs_msgs__action__CheckForObjects_Result result;
} eufs_msgs__action__CheckForObjects_GetResult_Response;

// Struct for a sequence of eufs_msgs__action__CheckForObjects_GetResult_Response.
typedef struct eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence
{
  eufs_msgs__action__CheckForObjects_GetResult_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__action__CheckForObjects_GetResult_Response__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'goal_id'
// already included above
// #include "unique_identifier_msgs/msg/detail/uuid__struct.h"
// Member 'feedback'
// already included above
// #include "eufs_msgs/action/detail/check_for_objects__struct.h"

// Struct defined in action/CheckForObjects in the package eufs_msgs.
typedef struct eufs_msgs__action__CheckForObjects_FeedbackMessage
{
  unique_identifier_msgs__msg__UUID goal_id;
  eufs_msgs__action__CheckForObjects_Feedback feedback;
} eufs_msgs__action__CheckForObjects_FeedbackMessage;

// Struct for a sequence of eufs_msgs__action__CheckForObjects_FeedbackMessage.
typedef struct eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence
{
  eufs_msgs__action__CheckForObjects_FeedbackMessage * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} eufs_msgs__action__CheckForObjects_FeedbackMessage__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // EUFS_MSGS__ACTION__DETAIL__CHECK_FOR_OBJECTS__STRUCT_H_
