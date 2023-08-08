// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/EKFErr.idl
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
#include "eufs_msgs/msg/detail/ekf_err__struct.h"
#include "eufs_msgs/msg/detail/ekf_err__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool std_msgs__msg__header__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * std_msgs__msg__header__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__ekf_err__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[30];
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
    assert(strncmp("eufs_msgs.msg._ekf_err.EKFErr", full_classname_dest, 29) == 0);
  }
  eufs_msgs__msg__EKFErr * ros_message = _ros_message;
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
  {  // gps_x_vel_err
    PyObject * field = PyObject_GetAttrString(_pymsg, "gps_x_vel_err");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->gps_x_vel_err = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // gps_y_vel_err
    PyObject * field = PyObject_GetAttrString(_pymsg, "gps_y_vel_err");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->gps_y_vel_err = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // imu_x_acc_err
    PyObject * field = PyObject_GetAttrString(_pymsg, "imu_x_acc_err");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->imu_x_acc_err = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // imu_y_acc_err
    PyObject * field = PyObject_GetAttrString(_pymsg, "imu_y_acc_err");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->imu_y_acc_err = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // imu_yaw_err
    PyObject * field = PyObject_GetAttrString(_pymsg, "imu_yaw_err");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->imu_yaw_err = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ekf_x_vel_var
    PyObject * field = PyObject_GetAttrString(_pymsg, "ekf_x_vel_var");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ekf_x_vel_var = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ekf_y_vel_var
    PyObject * field = PyObject_GetAttrString(_pymsg, "ekf_y_vel_var");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ekf_y_vel_var = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ekf_x_acc_var
    PyObject * field = PyObject_GetAttrString(_pymsg, "ekf_x_acc_var");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ekf_x_acc_var = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ekf_y_acc_var
    PyObject * field = PyObject_GetAttrString(_pymsg, "ekf_y_acc_var");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ekf_y_acc_var = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // ekf_yaw_var
    PyObject * field = PyObject_GetAttrString(_pymsg, "ekf_yaw_var");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->ekf_yaw_var = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__ekf_err__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of EKFErr */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._ekf_err");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "EKFErr");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__EKFErr * ros_message = (eufs_msgs__msg__EKFErr *)raw_ros_message;
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
  {  // gps_x_vel_err
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->gps_x_vel_err);
    {
      int rc = PyObject_SetAttrString(_pymessage, "gps_x_vel_err", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // gps_y_vel_err
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->gps_y_vel_err);
    {
      int rc = PyObject_SetAttrString(_pymessage, "gps_y_vel_err", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // imu_x_acc_err
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->imu_x_acc_err);
    {
      int rc = PyObject_SetAttrString(_pymessage, "imu_x_acc_err", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // imu_y_acc_err
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->imu_y_acc_err);
    {
      int rc = PyObject_SetAttrString(_pymessage, "imu_y_acc_err", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // imu_yaw_err
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->imu_yaw_err);
    {
      int rc = PyObject_SetAttrString(_pymessage, "imu_yaw_err", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ekf_x_vel_var
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ekf_x_vel_var);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ekf_x_vel_var", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ekf_y_vel_var
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ekf_y_vel_var);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ekf_y_vel_var", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ekf_x_acc_var
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ekf_x_acc_var);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ekf_x_acc_var", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ekf_y_acc_var
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ekf_y_acc_var);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ekf_y_acc_var", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ekf_yaw_var
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->ekf_yaw_var);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ekf_yaw_var", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
