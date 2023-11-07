// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from cmrdv_interfaces:msg/SimDataFrame.idl
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
#include "cmrdv_interfaces/msg/detail/sim_data_frame__struct.h"
#include "cmrdv_interfaces/msg/detail/sim_data_frame__functions.h"

ROSIDL_GENERATOR_C_IMPORT
bool eufs_msgs__msg__cone_array_with_covariance__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * eufs_msgs__msg__cone_array_with_covariance__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool sensor_msgs__msg__image__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * sensor_msgs__msg__image__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool sensor_msgs__msg__point_cloud2__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * sensor_msgs__msg__point_cloud2__convert_to_py(void * raw_ros_message);
ROSIDL_GENERATOR_C_IMPORT
bool sensor_msgs__msg__point_cloud2__convert_from_py(PyObject * _pymsg, void * _ros_message);
ROSIDL_GENERATOR_C_IMPORT
PyObject * sensor_msgs__msg__point_cloud2__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool cmrdv_interfaces__msg__sim_data_frame__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[50];
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
    assert(strncmp("cmrdv_interfaces.msg._sim_data_frame.SimDataFrame", full_classname_dest, 49) == 0);
  }
  cmrdv_interfaces__msg__SimDataFrame * ros_message = _ros_message;
  {  // gt_cones
    PyObject * field = PyObject_GetAttrString(_pymsg, "gt_cones");
    if (!field) {
      return false;
    }
    if (!eufs_msgs__msg__cone_array_with_covariance__convert_from_py(field, &ros_message->gt_cones)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // zed_left_img
    PyObject * field = PyObject_GetAttrString(_pymsg, "zed_left_img");
    if (!field) {
      return false;
    }
    if (!sensor_msgs__msg__image__convert_from_py(field, &ros_message->zed_left_img)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // vlp16_pts
    PyObject * field = PyObject_GetAttrString(_pymsg, "vlp16_pts");
    if (!field) {
      return false;
    }
    if (!sensor_msgs__msg__point_cloud2__convert_from_py(field, &ros_message->vlp16_pts)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }
  {  // zed_pts
    PyObject * field = PyObject_GetAttrString(_pymsg, "zed_pts");
    if (!field) {
      return false;
    }
    if (!sensor_msgs__msg__point_cloud2__convert_from_py(field, &ros_message->zed_pts)) {
      Py_DECREF(field);
      return false;
    }
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * cmrdv_interfaces__msg__sim_data_frame__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of SimDataFrame */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("cmrdv_interfaces.msg._sim_data_frame");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "SimDataFrame");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  cmrdv_interfaces__msg__SimDataFrame * ros_message = (cmrdv_interfaces__msg__SimDataFrame *)raw_ros_message;
  {  // gt_cones
    PyObject * field = NULL;
    field = eufs_msgs__msg__cone_array_with_covariance__convert_to_py(&ros_message->gt_cones);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "gt_cones", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // zed_left_img
    PyObject * field = NULL;
    field = sensor_msgs__msg__image__convert_to_py(&ros_message->zed_left_img);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "zed_left_img", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // vlp16_pts
    PyObject * field = NULL;
    field = sensor_msgs__msg__point_cloud2__convert_to_py(&ros_message->vlp16_pts);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "vlp16_pts", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // zed_pts
    PyObject * field = NULL;
    field = sensor_msgs__msg__point_cloud2__convert_to_py(&ros_message->zed_pts);
    if (!field) {
      return NULL;
    }
    {
      int rc = PyObject_SetAttrString(_pymessage, "zed_pts", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
