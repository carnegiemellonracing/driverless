#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "tinyxml2" for configuration "Release"
set_property(TARGET tinyxml2 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tinyxml2 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtinyxml2.so.6.2.0"
  IMPORTED_SONAME_RELEASE "libtinyxml2.so.6"
  )

list(APPEND _IMPORT_CHECK_TARGETS tinyxml2 )
list(APPEND _IMPORT_CHECK_FILES_FOR_tinyxml2 "${_IMPORT_PREFIX}/lib/libtinyxml2.so.6.2.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
