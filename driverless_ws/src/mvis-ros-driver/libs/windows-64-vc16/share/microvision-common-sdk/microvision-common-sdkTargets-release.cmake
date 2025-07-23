#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "microvision-common::sdk-core" for configuration "Release"
set_property(TARGET microvision-common::sdk-core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(microvision-common::sdk-core PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/microvision-common-sdk-core.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/microvision-common-sdk-core.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS microvision-common::sdk-core )
list(APPEND _IMPORT_CHECK_FILES_FOR_microvision-common::sdk-core "${_IMPORT_PREFIX}/lib/microvision-common-sdk-core.lib" "${_IMPORT_PREFIX}/bin/microvision-common-sdk-core.dll" )

# Import target "microvision-common::sdk-data" for configuration "Release"
set_property(TARGET microvision-common::sdk-data APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(microvision-common::sdk-data PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/microvision-common-sdk-data.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/microvision-common-sdk-data.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS microvision-common::sdk-data )
list(APPEND _IMPORT_CHECK_FILES_FOR_microvision-common::sdk-data "${_IMPORT_PREFIX}/lib/microvision-common-sdk-data.lib" "${_IMPORT_PREFIX}/bin/microvision-common-sdk-data.dll" )

# Import target "microvision-common::sdk-devices" for configuration "Release"
set_property(TARGET microvision-common::sdk-devices APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(microvision-common::sdk-devices PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/microvision-common-sdk-devices.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/microvision-common-sdk-devices.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS microvision-common::sdk-devices )
list(APPEND _IMPORT_CHECK_FILES_FOR_microvision-common::sdk-devices "${_IMPORT_PREFIX}/lib/microvision-common-sdk-devices.lib" "${_IMPORT_PREFIX}/bin/microvision-common-sdk-devices.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
