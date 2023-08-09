// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from eufs_msgs:msg/VehicleCommands.idl
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
#include "eufs_msgs/msg/detail/vehicle_commands__struct.h"
#include "eufs_msgs/msg/detail/vehicle_commands__functions.h"


ROSIDL_GENERATOR_C_EXPORT
bool eufs_msgs__msg__vehicle_commands__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[48];
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
    assert(strncmp("eufs_msgs.msg._vehicle_commands.VehicleCommands", full_classname_dest, 47) == 0);
  }
  eufs_msgs__msg__VehicleCommands * ros_message = _ros_message;
  {  // handshake
    PyObject * field = PyObject_GetAttrString(_pymsg, "handshake");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->handshake = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // ebs
    PyObject * field = PyObject_GetAttrString(_pymsg, "ebs");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->ebs = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // direction
    PyObject * field = PyObject_GetAttrString(_pymsg, "direction");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->direction = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // mission_status
    PyObject * field = PyObject_GetAttrString(_pymsg, "mission_status");
    if (!field) {
      return false;
    }
    assert(PyLong_Check(field));
    ros_message->mission_status = (int8_t)PyLong_AsLong(field);
    Py_DECREF(field);
  }
  {  // braking
    PyObject * field = PyObject_GetAttrString(_pymsg, "braking");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->braking = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }
  {  // torque
    PyObject * field = PyObject_GetAttrString(_pymsg, "torque");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->torque = PyFloat_AS_DOUBLE(field);
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
  {  // rpm
    PyObject * field = PyObject_GetAttrString(_pymsg, "rpm");
    if (!field) {
      return false;
    }
    assert(PyFloat_Check(field));
    ros_message->rpm = PyFloat_AS_DOUBLE(field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * eufs_msgs__msg__vehicle_commands__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of VehicleCommands */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("eufs_msgs.msg._vehicle_commands");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "VehicleCommands");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  eufs_msgs__msg__VehicleCommands * ros_message = (eufs_msgs__msg__VehicleCommands *)raw_ros_message;
  {  // handshake
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->handshake);
    {
      int rc = PyObject_SetAttrString(_pymessage, "handshake", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // ebs
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->ebs);
    {
      int rc = PyObject_SetAttrString(_pymessage, "ebs", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // direction
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->direction);
    {
      int rc = PyObject_SetAttrString(_pymessage, "direction", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // mission_status
    PyObject * field = NULL;
    field = PyLong_FromLong(ros_message->mission_status);
    {
      int rc = PyObject_SetAttrString(_pymessage, "mission_status", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // braking
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->braking);
    {
      int rc = PyObject_SetAttrString(_pymessage, "braking", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }
  {  // torque
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->torque);
    {
      int rc = PyObject_SetAttrString(_pymessage, "torque", field);
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
  {  // rpm
    PyObject * field = NULL;
    field = PyFloat_FromDouble(ros_message->rpm);
    {
      int rc = PyObject_SetAttrString(_pymessage, "rpm", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
