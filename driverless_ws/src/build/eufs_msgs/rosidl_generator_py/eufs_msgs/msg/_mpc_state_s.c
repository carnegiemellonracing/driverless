// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/MPCState.idl
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
#include "eufs_msgs/msg/detail/mpc_state__struct.h"
#include "eufs_msgs/msg/detail/mpc_state__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__mpc_state__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[34];
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
    assert(strncmp("eufs_msgs.msg._mpc_state.MPCState", full_classname_dest, 33) == 0);
  }
  eufs_msgs__msg__MPCState * ros_message = _ros_message;
  {  // exitflag
    PyObject * field = PyObject_GetAttrString(_pymsg, "exitflag");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->exitflag = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // iterations
    PyObject * field = PyObject_GetAttrString(_pymsg, "iterations");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->iterations = (uint8_t)PyLong_AsUnsignedLong(field);
    Py_DECREF(field);
  }
  {  // solve_time
    PyObject * field = PyObject_GetAttrString(_pymsg, "solve_time");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->solve_time = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // cost
    PyObject * field = PyObject_GetAttrString(_pymsg, "cost");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->cost = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__mpc_state__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of MPCState */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._mpc_state");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "MPCState");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__MPCState * ros_message = (eufs_msgs__msg__MPCState *)raw_ros_message;
  {  // exitflag
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->exitflag);
    {
      int rc = PyObject_SetAttrString(_pymessage, "exitflag", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // iterations
    PyObject * field = NULL;
    field = PyLong_FromUnsignedLong(ros_message->iterations);
    {
      int rc = PyObject_SetAttrString(_pymessage, "iterations", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // solve_time
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->solve_time);
    {
      int rc = PyObject_SetAttrString(_pymessage, "solve_time", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // cost
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->cost);
    {
      int rc = PyObject_SetAttrString(_pymessage, "cost", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
