// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/PathIntegralParams.idl
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
#include "eufs_msgs/msg/detail/path_integral_params__struct.h"
#include "eufs_msgs/msg/detail/path_integral_params__functions.h"

#include "rosidl_runtime_c/string.h"
#include "rosidl_runtime_c/string_functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__path_integral_params__convert_from_py(PyObject * _pymsg, void * _ros_message)
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
    assert(strncmp("eufs_msgs.msg._path_integral_params.PathIntegralParams", full_classname_dest, 54) == 0);
  }
  eufs_msgs__msg__PathIntegralParams * ros_message = _ros_message;
  {  // hz
    PyObject * field = PyObject_GetAttrString(_pymsg, "hz");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->hz = PyLong_AsLongLong(field);
    Py_DECREF(field);
  }
  {  // num_timesteps
    PyObject * field = PyObject_GetAttrString(_pymsg, "num_timesteps");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->num_timesteps = PyLong_AsLongLong(field);
    Py_DECREF(field);
  }
  {  // num_iters
    PyObject * field = PyObject_GetAttrString(_pymsg, "num_iters");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->num_iters = PyLong_AsLongLong(field);
    Py_DECREF(field);
  }
  {  // gamma
    PyObject * field = PyObject_GetAttrString(_pymsg, "gamma");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->gamma = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // init_steering
    PyObject * field = PyObject_GetAttrString(_pymsg, "init_steering");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->init_steering = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // init_throttle
    PyObject * field = PyObject_GetAttrString(_pymsg, "init_throttle");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->init_throttle = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // steering_var
    PyObject * field = PyObject_GetAttrString(_pymsg, "steering_var");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->steering_var = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // throttle_var
    PyObject * field = PyObject_GetAttrString(_pymsg, "throttle_var");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->throttle_var = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // max_throttle
    PyObject * field = PyObject_GetAttrString(_pymsg, "max_throttle");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->max_throttle = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // speed_coefficient
    PyObject * field = PyObject_GetAttrString(_pymsg, "speed_coefficient");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->speed_coefficient = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // track_coefficient
    PyObject * field = PyObject_GetAttrString(_pymsg, "track_coefficient");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->track_coefficient = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // max_slip_angle
    PyObject * field = PyObject_GetAttrString(_pymsg, "max_slip_angle");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->max_slip_angle = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // track_slop
    PyObject * field = PyObject_GetAttrString(_pymsg, "track_slop");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->track_slop = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // crash_coeff
    PyObject * field = PyObject_GetAttrString(_pymsg, "crash_coeff");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->crash_coeff = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // map_path
    PyObject * field = PyObject_GetAttrString(_pymsg, "map_path");
    if (!field) {
      return false;
    }
    assert(PyUnicode_Check(field));
    PyObject * encoded_field = PyUnicode_AsUTF8String(field);
    if (!encoded_field) {
      Py_DECREF(field);
      return false;
    }
    rosidl_runtime_c__String__assign(&ros_message->map_path, PyBytes_AS_STRING(encoded_field));
    Py_DECREF(encoded_field);
    Py_DECREF(field);
  }
  {  // desired_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "desired_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->desired_speed = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__path_integral_params__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of PathIntegralParams */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._path_integral_params");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "PathIntegralParams");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__PathIntegralParams * ros_message = (eufs_msgs__msg__PathIntegralParams *)raw_ros_message;
  {  // hz
    PyObject * field = NULL;
    field = PyLong_FromLongLong(ros_message->hz);
    {
      int rc = PyObject_SetAttrString(_pymessage, "hz", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // num_timesteps
    PyObject * field = NULL;
    field = PyLong_FromLongLong(ros_message->num_timesteps);
    {
      int rc = PyObject_SetAttrString(_pymessage, "num_timesteps", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // num_iters
    PyObject * field = NULL;
    field = PyLong_FromLongLong(ros_message->num_iters);
    {
      int rc = PyObject_SetAttrString(_pymessage, "num_iters", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // gamma
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->gamma);
    {
      int rc = PyObject_SetAttrString(_pymessage, "gamma", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // init_steering
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->init_steering);
    {
      int rc = PyObject_SetAttrString(_pymessage, "init_steering", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // init_throttle
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->init_throttle);
    {
      int rc = PyObject_SetAttrString(_pymessage, "init_throttle", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // steering_var
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->steering_var);
    {
      int rc = PyObject_SetAttrString(_pymessage, "steering_var", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // throttle_var
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->throttle_var);
    {
      int rc = PyObject_SetAttrString(_pymessage, "throttle_var", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // max_throttle
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->max_throttle);
    {
      int rc = PyObject_SetAttrString(_pymessage, "max_throttle", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // speed_coefficient
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->speed_coefficient);
    {
      int rc = PyObject_SetAttrString(_pymessage, "speed_coefficient", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // track_coefficient
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->track_coefficient);
    {
      int rc = PyObject_SetAttrString(_pymessage, "track_coefficient", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // max_slip_angle
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->max_slip_angle);
    {
      int rc = PyObject_SetAttrString(_pymessage, "max_slip_angle", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // track_slop
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->track_slop);
    {
      int rc = PyObject_SetAttrString(_pymessage, "track_slop", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // crash_coeff
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->crash_coeff);
    {
      int rc = PyObject_SetAttrString(_pymessage, "crash_coeff", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // map_path
    PyObject * field = NULL;
    field = PyUnicode_DecodeUTF8(
      ros_message->map_path.data,
      strlen(ros_message->map_path.data),
      "replace");
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "map_path", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // desired_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->desired_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "desired_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
