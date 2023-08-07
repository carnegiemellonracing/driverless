// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/ChassisCommand.idl
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
#include "eufs_msgs/msg/detail/chassis_command__struct.h"
#include "eufs_msgs/msg/detail/chassis_command__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__header__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__chassis_command__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[46];
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
    assert(strncmp("eufs_msgs.msg._chassis_command.ChassisCommand", full_classname_dest, 45) == 0);
  }
  eufs_msgs__msg__ChassisCommand * ros_message = _ros_message;
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
  {  // sender
    PyObject * field = PyObject_GetAttrString(_pymsg, "sender");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->sender, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // throttle
    PyObject * field = PyObject_GetAttrString(_pymsg, "throttle");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->throttle = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // steering
    PyObject * field = PyObject_GetAttrString(_pymsg, "steering");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->steering = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // front_brake
    PyObject * field = PyObject_GetAttrString(_pymsg, "front_brake");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->front_brake = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__chassis_command__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of ChassisCommand */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._chassis_command");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "ChassisCommand");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__ChassisCommand * ros_message = (eufs_msgs__msg__ChassisCommand *)raw_ros_message;
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
  {  // sender
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->sender.data,
      strlen(ros_message->sender.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "sender", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // throttle
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->throttle);
    {
      int rc = PyObject_SetAttrString(_pymessage, "throttle", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
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
  {  // front_brake
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->front_brake);
    {
      int rc = PyObject_SetAttrString(_pymessage, "front_brake", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
