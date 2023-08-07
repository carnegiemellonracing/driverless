// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/PathIntegralTiming.idl
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
#include "eufs_msgs/msg/detail/path_integral_timing__struct.h"
#include "eufs_msgs/msg/detail/path_integral_timing__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__header__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__path_integral_timing__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[55];
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
    assert(strncmp("eufs_msgs.msg._path_integral_timing.PathIntegralTiming", full_classname_dest, 54) == 0);
  }
  eufs_msgs__msg__PathIntegralTiming * ros_message = _ros_message;
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
  {  // average_time_between_poses
    PyObject * field = PyObject_GetAttrString(_pymsg, "average_time_between_poses");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->average_time_between_poses = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // average_optimization_cycle_time
    PyObject * field = PyObject_GetAttrString(_pymsg, "average_optimization_cycle_time");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->average_optimization_cycle_time = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // average_sleep_time
    PyObject * field = PyObject_GetAttrString(_pymsg, "average_sleep_time");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->average_sleep_time = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__path_integral_timing__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of PathIntegralTiming */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._path_integral_timing");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "PathIntegralTiming");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__PathIntegralTiming * ros_message = (eufs_msgs__msg__PathIntegralTiming *)raw_ros_message;
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
  {  // average_time_between_poses
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->average_time_between_poses);
    {
      int rc = PyObject_SetAttrString(_pymessage, "average_time_between_poses", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // average_optimization_cycle_time
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->average_optimization_cycle_time);
    {
      int rc = PyObject_SetAttrString(_pymessage, "average_optimization_cycle_time", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // average_sleep_time
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->average_sleep_time);
    {
      int rc = PyObject_SetAttrString(_pymessage, "average_sleep_time", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
