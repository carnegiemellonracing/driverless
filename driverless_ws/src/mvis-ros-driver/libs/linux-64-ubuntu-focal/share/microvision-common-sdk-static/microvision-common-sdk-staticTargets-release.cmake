#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "microvision-common::sdk-core" for configuration "Release"
set_property(TARGET microvision-common::sdk-core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(microvision-common::sdk-core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmicrovision-common-sdk-core.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS microvision-common::sdk-core )
list(APPEND _IMPORT_CHECK_FILES_FOR_microvision-common::sdk-core "${_IMPORT_PREFIX}/lib/libmicrovision-common-sdk-core.a" )

# Import target "microvision-common::sdk-data" for configuration "Release"
set_property(TARGET microvision-common::sdk-data APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(microvision-common::sdk-data PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmicrovision-common-sdk-data.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS microvision-common::sdk-data )
list(APPEND _IMPORT_CHECK_FILES_FOR_microvision-common::sdk-data "${_IMPORT_PREFIX}/lib/libmicrovision-common-sdk-data.a" )

# Import target "microvision-common::sdk-devices" for configuration "Release"
set_property(TARGET microvision-common::sdk-devices APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(microvision-common::sdk-devices PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmicrovision-common-sdk-devices.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS microvision-common::sdk-devices )
list(APPEND _IMPORT_CHECK_FILES_FOR_microvision-common::sdk-devices "${_IMPORT_PREFIX}/lib/libmicrovision-common-sdk-devices.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
