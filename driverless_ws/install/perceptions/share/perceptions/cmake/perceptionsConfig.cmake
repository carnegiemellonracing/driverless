# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_perceptions_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED perceptions_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(perceptions_FOUND FALSE)
  elseif(NOT perceptions_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(perceptions_FOUND FALSE)
  endif()
  return()
endif()
set(_perceptions_CONFIG_INCLUDED TRUE)

# output package information
if(NOT perceptions_FIND_QUIETLY)
  message(STATUS "Found perceptions: 0.0.0 (${perceptions_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'perceptions' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT perceptions_DEPRECATED_QUIET)
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(perceptions_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${perceptions_DIR}/${_extra}")
endforeach()
