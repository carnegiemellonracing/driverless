#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "turbojpeg" for configuration "Release"
set_property(TARGET turbojpeg APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(turbojpeg PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/turbojpeg.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/turbojpeg.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS turbojpeg )
list(APPEND _IMPORT_CHECK_FILES_FOR_turbojpeg "${_IMPORT_PREFIX}/lib/turbojpeg.lib" "${_IMPORT_PREFIX}/bin/turbojpeg.dll" )

# Import target "tjbench" for configuration "Release"
set_property(TARGET tjbench APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(tjbench PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/tjbench.exe"
  )

list(APPEND _IMPORT_CHECK_TARGETS tjbench )
list(APPEND _IMPORT_CHECK_FILES_FOR_tjbench "${_IMPORT_PREFIX}/bin/tjbench.exe" )

# Import target "turbojpeg-static" for configuration "Release"
set_property(TARGET turbojpeg-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(turbojpeg-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/turbojpeg-static.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS turbojpeg-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_turbojpeg-static "${_IMPORT_PREFIX}/lib/turbojpeg-static.lib" )

# Import target "jpeg-static" for configuration "Release"
set_property(TARGET jpeg-static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(jpeg-static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/jpeg-static.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS jpeg-static )
list(APPEND _IMPORT_CHECK_FILES_FOR_jpeg-static "${_IMPORT_PREFIX}/lib/jpeg-static.lib" )

# Import target "rdjpgcom" for configuration "Release"
set_property(TARGET rdjpgcom APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(rdjpgcom PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/rdjpgcom.exe"
  )

list(APPEND _IMPORT_CHECK_TARGETS rdjpgcom )
list(APPEND _IMPORT_CHECK_FILES_FOR_rdjpgcom "${_IMPORT_PREFIX}/bin/rdjpgcom.exe" )

# Import target "wrjpgcom" for configuration "Release"
set_property(TARGET wrjpgcom APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(wrjpgcom PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/wrjpgcom.exe"
  )

list(APPEND _IMPORT_CHECK_TARGETS wrjpgcom )
list(APPEND _IMPORT_CHECK_FILES_FOR_wrjpgcom "${_IMPORT_PREFIX}/bin/wrjpgcom.exe" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
