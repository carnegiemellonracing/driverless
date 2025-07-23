#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "microvision-common-logging::logging" for configuration "Release"
set_property(TARGET microvision-common-logging::logging APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(microvision-common-logging::logging PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmicrovision-common-logging.so.8.6.0"
  IMPORTED_SONAME_RELEASE "libmicrovision-common-logging.so.8"
  )

list(APPEND _IMPORT_CHECK_TARGETS microvision-common-logging::logging )
list(APPEND _IMPORT_CHECK_FILES_FOR_microvision-common-logging::logging "${_IMPORT_PREFIX}/lib/libmicrovision-common-logging.so.8.6.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
