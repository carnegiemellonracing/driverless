// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/LapStats.idl
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
#include "eufs_msgs/msg/detail/lap_stats__struct.h"
#include "eufs_msgs/msg/detail/lap_stats__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__header__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__lap_stats__convert_from_py(PyObject * _pymsg, void * _ros_message)
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
    assert(strncmp("eufs_msgs.msg._lap_stats.LapStats", full_classname_dest, 33) == 0);
  }
  eufs_msgs__msg__LapStats * ros_message = _ros_message;
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
  {  // lap_number
    PyObject * field = PyObject_GetAttrString(_pymsg, "lap_number");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->lap_number = PyLong_AsLongLong(field);
    Py_DECREF(field);
  }
  {  // lap_time
    PyObject * field = PyObject_GetAttrString(_pymsg, "lap_time");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->lap_time = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // avg_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "avg_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->avg_speed = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // max_speed
    PyObject * field = PyObject_GetAttrString(_pymsg, "max_speed");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->max_speed = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // speed_var
    PyObject * field = PyObject_GetAttrString(_pymsg, "speed_var");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->speed_var = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // max_slip
    PyObject * field = PyObject_GetAttrString(_pymsg, "max_slip");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->max_slip = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // max_lateral_accel
    PyObject * field = PyObject_GetAttrString(_pymsg, "max_lateral_accel");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->max_lateral_accel = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // normalized_deviation_mse
    PyObject * field = PyObject_GetAttrString(_pymsg, "normalized_deviation_mse");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->normalized_deviation_mse = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // deviation_var
    PyObject * field = PyObject_GetAttrString(_pymsg, "deviation_var");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->deviation_var = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // max_deviation
    PyObject * field = PyObject_GetAttrString(_pymsg, "max_deviation");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->max_deviation = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__lap_stats__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of LapStats */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._lap_stats");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "LapStats");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__LapStats * ros_message = (eufs_msgs__msg__LapStats *)raw_ros_message;
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
  {  // lap_number
    PyObject * field = NULL;
    field = PyLong_FromLongLong(ros_message->lap_number);
    {
      int rc = PyObject_SetAttrString(_pymessage, "lap_number", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // lap_time
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->lap_time);
    {
      int rc = PyObject_SetAttrString(_pymessage, "lap_time", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // avg_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->avg_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "avg_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // max_speed
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->max_speed);
    {
      int rc = PyObject_SetAttrString(_pymessage, "max_speed", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // speed_var
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->speed_var);
    {
      int rc = PyObject_SetAttrString(_pymessage, "speed_var", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // max_slip
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->max_slip);
    {
      int rc = PyObject_SetAttrString(_pymessage, "max_slip", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // max_lateral_accel
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->max_lateral_accel);
    {
      int rc = PyObject_SetAttrString(_pymessage, "max_lateral_accel", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // normalized_deviation_mse
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->normalized_deviation_mse);
    {
      int rc = PyObject_SetAttrString(_pymessage, "normalized_deviation_mse", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // deviation_var
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->deviation_var);
    {
      int rc = PyObject_SetAttrString(_pymessage, "deviation_var", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // max_deviation
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->max_deviation);
    {
      int rc = PyObject_SetAttrString(_pymessage, "max_deviation", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
