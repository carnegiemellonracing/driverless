// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from cmrdv_interfaces:msg/ControlAction.idl
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
#include "cmrdv_interfaces/msg/detail/control_action__struct.h"
#include "cmrdv_interfaces/msg/detail/control_action__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool cmrdv_interfaces__msg__control_action__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[51];
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
    assert(strncmp("cmrdv_interfaces.msg._control_action.ControlAction", full_classname_dest, 50) == 0);
  }
  cmrdv_interfaces__msg__ControlAction * ros_message = _ros_message;
  {  // wheel_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "wheel_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->wheel_speed = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // swangle
    PyObject * field = PyObject_GetAttrString(_pymsg, "swangle");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->swangle = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * cmrdv_interfaces__msg__control_action__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of ControlAction */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("cmrdv_interfaces.msg._control_action");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "ControlAction");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  cmrdv_interfaces__msg__ControlAction * ros_message = (cmrdv_interfaces__msg__ControlAction *)raw_ros_message;
  {  // wheel_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->wheel_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "wheel_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // swangle
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->swangle);
    {
      int rc = PyObject_SetAttrString(_pymessage, "swangle", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
