// generated from
// rosidl_typesupport_c/resource/rosidl_typesupport_c__visibility_control.h.in
// generated code does not contain a copyright notice

#ifndef CMRDV_INTERFACES__MSG__ROSIDL_TYPESUPPORT_C__VISIBILITY_CONTROL_H_
#define CMRDV_INTERFACES__MSG__ROSIDL_TYPESUPPORT_C__VISIBILITY_CONTROL_H_

#ifdef __cplusplus
extern "C"
{
#endif

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define ROSIDL_TYPESUPPORT_C_EXPORT_cmrdv_interfaces __attribute__ ((dllexport))
    #define ROSIDL_TYPESUPPORT_C_IMPORT_cmrdv_interfaces __attribute__ ((dllimport))
  #else
    #define ROSIDL_TYPESUPPORT_C_EXPORT_cmrdv_interfaces __declspec(dllexport)
    #define ROSIDL_TYPESUPPORT_C_IMPORT_cmrdv_interfaces __declspec(dllimport)
  #endif
  #ifdef ROSIDL_TYPESUPPORT_C_BUILDING_DLL_cmrdv_interfaces
    #define ROSIDL_TYPESUPPORT_C_PUBLIC_cmrdv_interfaces ROSIDL_TYPESUPPORT_C_EXPORT_cmrdv_interfaces
  #else
    #define ROSIDL_TYPESUPPORT_C_PUBLIC_cmrdv_interfaces ROSIDL_TYPESUPPORT_C_IMPORT_cmrdv_interfaces
  #endif
#else
  #define ROSIDL_TYPESUPPORT_C_EXPORT_cmrdv_interfaces __attribute__ ((visibility("default")))
  #define ROSIDL_TYPESUPPORT_C_IMPORT_cmrdv_interfaces
  #if __GNUC__ >= 4
    #define ROSIDL_TYPESUPPORT_C_PUBLIC_cmrdv_interfaces __attribute__ ((visibility("default")))
  #else
    #define ROSIDL_TYPESUPPORT_C_PUBLIC_cmrdv_interfaces
  #endif
#endif

#ifdef __cplusplus
}
#endif

#endif  // CMRDV_INTERFACES__MSG__ROSIDL_TYPESUPPORT_C__VISIBILITY_CONTROL_H_
