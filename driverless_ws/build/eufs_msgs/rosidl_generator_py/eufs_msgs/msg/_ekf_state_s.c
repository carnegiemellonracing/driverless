// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/EKFState.idl
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
#include "eufs_msgs/msg/detail/ekf_state__struct.h"
#include "eufs_msgs/msg/detail/ekf_state__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__ekf_state__convert_from_py(PyObject * _pymsg, void * _ros_message)
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
    assert(strncmp("eufs_msgs.msg._ekf_state.EKFState", full_classname_dest, 33) == 0);
  }
  eufs_msgs__msg__EKFState * ros_message = _ros_message;
  {  // gps_received
    PyObject * field = PyObject_GetAttrString(_pymsg, "gps_received");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->gps_received = (Py_True == field);
    Py_DECREF(field);
  }
  {  // imu_received
    PyObject * field = PyObject_GetAttrString(_pymsg, "imu_received");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->imu_received = (Py_True == field);
    Py_DECREF(field);
  }
  {  // wheel_odom_received
    PyObject * field = PyObject_GetAttrString(_pymsg, "wheel_odom_received");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->wheel_odom_received = (Py_True == field);
    Py_DECREF(field);
  }
  {  // ekf_odom_received
    PyObject * field = PyObject_GetAttrString(_pymsg, "ekf_odom_received");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->ekf_odom_received = (Py_True == field);
    Py_DECREF(field);
  }
  {  // ekf_accel_received
    PyObject * field = PyObject_GetAttrString(_pymsg, "ekf_accel_received");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->ekf_accel_received = (Py_True == field);
    Py_DECREF(field);
  }
  {  // currently_over_covariance_limit
    PyObject * field = PyObject_GetAttrString(_pymsg, "currently_over_covariance_limit");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->currently_over_covariance_limit = (Py_True == field);
    Py_DECREF(field);
  }
  {  // consecutive_turns_over_covariance_limit
    PyObject * field = PyObject_GetAttrString(_pymsg, "consecutive_turns_over_covariance_limit");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->consecutive_turns_over_covariance_limit = (Py_True == field);
    Py_DECREF(field);
  }
  {  // recommends_failure
    PyObject * field = PyObject_GetAttrString(_pymsg, "recommends_failure");
    if (!field) {
      return false;
    }
    assert(PyBool_Check(field));
    ros_message->recommends_failure = (Py_True == field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__ekf_state__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of EKFState */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._ekf_state");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "EKFState");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__EKFState * ros_message = (eufs_msgs__msg__EKFState *)raw_ros_message;
  {  // gps_received
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->gps_received ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "gps_received", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // imu_received
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->imu_received ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "imu_received", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // wheel_odom_received
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->wheel_odom_received ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "wheel_odom_received", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ekf_odom_received
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->ekf_odom_received ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ekf_odom_received", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ekf_accel_received
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->ekf_accel_received ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ekf_accel_received", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // currently_over_covariance_limit
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->currently_over_covariance_limit ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "currently_over_covariance_limit", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // consecutive_turns_over_covariance_limit
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->consecutive_turns_over_covariance_limit ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "consecutive_turns_over_covariance_limit", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // recommends_failure
    PyObject * field = NULL;
    field = PyBool_FromLong(ros_message->recommends_failure ? 1 : 0);
    {
      int rc = PyObject_SetAttrString(_pymessage, "recommends_failure", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
