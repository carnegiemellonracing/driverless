// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/WheelSpeeds.idl
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
#include "eufs_msgs/msg/detail/wheel_speeds__struct.h"
#include "eufs_msgs/msg/detail/wheel_speeds__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__wheel_speeds__convert_from_py(PyObject * _pymsg, void * _ros_message)
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
    assert(strncmp("eufs_msgs.msg._wheel_speeds.WheelSpeeds", full_classname_dest, 39) == 0);
  }
  eufs_msgs__msg__WheelSpeeds * ros_message = _ros_message;
  {  // steering
    PyObject * field = PyObject_GetAttrString(_pymsg, "steering");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->steering = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // lf_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "lf_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->lf_speed = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // rf_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "rf_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->rf_speed = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // lb_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "lb_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->lb_speed = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // rb_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "rb_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->rb_speed = (float)PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__wheel_speeds__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of WheelSpeeds */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._wheel_speeds");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "WheelSpeeds");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__WheelSpeeds * ros_message = (eufs_msgs__msg__WheelSpeeds *)raw_ros_message;
  {  // steering
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->steering);
    {
      int rc = PyObject_SetAttrString(_pymessage, "steering", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // lf_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->lf_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "lf_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rf_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->rf_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rf_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // lb_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->lb_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "lb_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // rb_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->rb_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rb_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
