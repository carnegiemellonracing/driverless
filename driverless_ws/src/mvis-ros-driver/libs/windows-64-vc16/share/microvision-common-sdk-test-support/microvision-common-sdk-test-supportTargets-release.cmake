#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "microvision-common::sdk-test-support" for configuration "Release"
set_property(TARGET microvision-common::sdk-test-support APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(microvision-common::sdk-test-support PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/tests/lib/microvision-common-sdk-test-support.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/tests/bin/microvision-common-sdk-test-support.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS microvision-common::sdk-test-support )
list(APPEND _IMPORT_CHECK_FILES_FOR_microvision-common::sdk-test-support "${_IMPORT_PREFIX}/tests/lib/microvision-common-sdk-test-support.lib" "${_IMPORT_PREFIX}/tests/bin/microvision-common-sdk-test-support.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
