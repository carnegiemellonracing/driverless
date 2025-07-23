#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "microvision-common-plugin_framework::plugin_loader" for configuration "Release"
set_property(TARGET microvision-common-plugin_framework::plugin_loader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(microvision-common-plugin_framework::plugin_loader PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmicrovision-common-pluginloader.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS microvision-common-plugin_framework::plugin_loader )
list(APPEND _IMPORT_CHECK_FILES_FOR_microvision-common-plugin_framework::plugin_loader "${_IMPORT_PREFIX}/lib/libmicrovision-common-pluginloader.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
