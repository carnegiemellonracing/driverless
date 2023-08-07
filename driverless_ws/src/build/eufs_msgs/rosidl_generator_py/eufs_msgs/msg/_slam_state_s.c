// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/SLAMState.idl
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
#include "eufs_msgs/msg/detail/slam_state__struct.h"
#include "eufs_msgs/msg/detail/slam_state__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__slam_state__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[36];
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
    assert(strncmp("eufs_msgs.msg._slam_state.SLAMState", full_classname_dest, 35) == 0);
  }
  eufs_msgs__msg__SLAMState * ros_message = _ros_message;
  {  // loop_closed
    PyObject * field = PyObject_GetAttrString(_pymsg, "loop_closed");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->loop_closed = (Py_True == field);
    Py_DECREF(field);
  }
  {  // laps
    PyObject * field = PyObject_GetAttrString(_pymsg, "laps");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->laps = (int16_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // status
    PyObject * field = PyObject_GetAttrString(_pymsg, "status");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->status, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // state
    PyObject * field = PyObject_GetAttrString(_pymsg, "state");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->state = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__slam_state__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of SLAMState */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._slam_state");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "SLAMState");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__SLAMState * ros_message = (eufs_msgs__msg__SLAMState *)raw_ros_message;
  {  // loop_closed
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->loop_closed ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "loop_closed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // laps
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->laps);
    {
      int rc = PyObject_SetAttrString(_pymessage, "laps", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // status
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->status.data,
      strlen(ros_message->status.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "status", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // state
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->state);
    {
      int rc = PyObject_SetAttrString(_pymessage, "state", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
