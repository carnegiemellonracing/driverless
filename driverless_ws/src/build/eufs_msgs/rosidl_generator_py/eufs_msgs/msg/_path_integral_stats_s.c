// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/PathIntegralStats.idl
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
#include "eufs_msgs/msg/detail/path_integral_stats__struct.h"
#include "eufs_msgs/msg/detail/path_integral_stats__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__header__convert_to_py(void * raw_ros_message);
bool eufs_msgs__msg__path_integral_params__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * eufs_msgs__msg__path_integral_params__convert_to_py(void * raw_ros_message);
bool eufs_msgs__msg__lap_stats__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * eufs_msgs__msg__lap_stats__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__path_integral_stats__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[53];
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
    assert(strncmp("eufs_msgs.msg._path_integral_stats.PathIntegralStats", full_classname_dest, 52) == 0);
  }
  eufs_msgs__msg__PathIntegralStats * ros_message = _ros_message;
  {  // header
    PyObject * field = PyObject_GetAttrString(_pymsg, "header");
    if (!field) {
      return false;
    }
    if (!std_msgs__msg__header__convert_from_py(field, &ros_message->header)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // tag
    PyObject * field = PyObject_GetAttrString(_pymsg, "tag");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->tag, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // params
    PyObject * field = PyObject_GetAttrString(_pymsg, "params");
    if (!field) {
      return false;
    }
    if (!eufs_msgs__msg__path_integral_params__convert_from_py(field, &ros_message->params)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // stats
    PyObject * field = PyObject_GetAttrString(_pymsg, "stats");
    if (!field) {
      return false;
    }
    if (!eufs_msgs__msg__lap_stats__convert_from_py(field, &ros_message->stats)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__path_integral_stats__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of PathIntegralStats */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._path_integral_stats");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "PathIntegralStats");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__PathIntegralStats * ros_message = (eufs_msgs__msg__PathIntegralStats *)raw_ros_message;
  {  // header
    PyObject * field = NULL;
    field = std_msgs__msg__header__convert_to_py(&ros_message->header);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "header", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // tag
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->tag.data,
      strlen(ros_message->tag.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "tag", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // params
    PyObject * field = NULL;
    field = eufs_msgs__msg__path_integral_params__convert_to_py(&ros_message->params);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "params", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // stats
    PyObject * field = NULL;
    field = eufs_msgs__msg__lap_stats__convert_to_py(&ros_message->stats);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "stats", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
