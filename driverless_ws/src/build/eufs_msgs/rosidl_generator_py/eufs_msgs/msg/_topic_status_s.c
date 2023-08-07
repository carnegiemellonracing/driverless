// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/TopicStatus.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "eufs_msgs/msg/detail/topic_status__struct.h"
#include "eufs_msgs/msg/detail/topic_status__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__topic_status__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[40];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("eufs_msgs.msg._topic_status.TopicStatus", full_classname_dest, 39) == 0);
  }
  eufs_msgs__msg__TopicStatus * ros_message = _ros_message;
  {  // topic
    PyObject * field = PyObject_GetAttrString(_pymsg, "topic");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->topic, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // description
    PyObject * field = PyObject_GetAttrString(_pymsg, "description");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->description, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // group
    PyObject * field = PyObject_GetAttrString(_pymsg, "group");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->group, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // trigger_ebs
    PyObject * field = PyObject_GetAttrString(_pymsg, "trigger_ebs");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->trigger_ebs = (Py_True == field);
    Py_DECREF(field);
  }
  {  // log_level
    PyObject * field = PyObject_GetAttrString(_pymsg, "log_level");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->log_level, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // status
    PyObject * field = PyObject_GetAttrString(_pymsg, "status");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->status = (uint16_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__topic_status__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of TopicStatus */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._topic_status");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "TopicStatus");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__TopicStatus * ros_message = (eufs_msgs__msg__TopicStatus *)raw_ros_message;
  {  // topic
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->topic.data,
      strlen(ros_message->topic.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "topic", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // description
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->description.data,
      strlen(ros_message->description.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "description", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // group
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->group.data,
      strlen(ros_message->group.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "group", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // trigger_ebs
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->trigger_ebs ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "trigger_ebs", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // log_level
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->log_level.data,
      strlen(ros_message->log_level.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "log_level", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // status
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->status);
    {
      int rc = PyObject_SetAttrString(_pymessage, "status", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
