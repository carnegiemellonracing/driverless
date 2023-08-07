# generated from
# ament_cmake_core/cmake/symlink_install/ament_cmake_symlink_install.cmake.in

# create empty symlink install manifest before starting install step
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/symlink_install_manifest.txt")

#
# Reimplement CMake install(DIRECTORY) command to use symlinks instead of
# copying resources.
#
# :param cmake_current_source_dir: The CMAKE_CURRENT_SOURCE_DIR when install
#   was invoked
# :type cmake_current_source_dir: string
# :param ARGN: the same arguments as the CMake install command.
# :type ARGN: various
#
function(ament_cmake_symlink_install_directory cmake_current_source_dir)
  cmake_parse_arguments(ARG "OPTIONAL" "DESTINATION" "DIRECTORY;PATTERN;PATTERN_EXCLUDE" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "ament_cmake_symlink_install_directory() called with "
      "unused/unsupported arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  # make destination absolute path and ensure that it exists
  if(NOT IS_ABSOLUTE "${ARG_DESTINATION}")
    set(ARG_DESTINATION "/driverless/driverless_ws/install/eufs_msgs/${ARG_DESTINATION}")
  endif()
  if(NOT EXISTS "${ARG_DESTINATION}")
    file(MAKE_DIRECTORY "${ARG_DESTINATION}")
  endif()

  # default pattern to include
  if(NOT ARG_PATTERN)
    set(ARG_PATTERN "*")
  endif()

  # iterate over directories
  foreach(dir ${ARG_DIRECTORY})
    # make dir an absolute path
    if(NOT IS_ABSOLUTE "${dir}")
      set(dir "${cmake_current_source_dir}/${dir}")
    endif()

    if(EXISTS "${dir}")
      # if directory has no trailing slash
      # append folder name to destination
      set(destination "${ARG_DESTINATION}")
      string(LENGTH "${dir}" length)
      math(EXPR offset "${length} - 1")
      string(SUBSTRING "${dir}" ${offset} 1 dir_last_char)
      if(NOT dir_last_char STREQUAL "/")
        get_filename_component(destination_name "${dir}" NAME)
        set(destination "${destination}/${destination_name}")
      else()
        # remove trailing slash
        string(SUBSTRING "${dir}" 0 ${offset} dir)
      endif()

      # glob recursive files
      set(relative_files "")
      foreach(pattern ${ARG_PATTERN})
        file(
          GLOB_RECURSE
          include_files
          RELATIVE "${dir}"
          "${dir}/${pattern}"
        )
        if(NOT include_files STREQUAL "")
          list(APPEND relative_files ${include_files})
        endif()
      endforeach()
      foreach(pattern ${ARG_PATTERN_EXCLUDE})
        file(
          GLOB_RECURSE
          exclude_files
          RELATIVE "${dir}"
          "${dir}/${pattern}"
        )
        if(NOT exclude_files STREQUAL "")
          list(REMOVE_ITEM relative_files ${exclude_files})
        endif()
      endforeach()
      list(SORT relative_files)

      foreach(relative_file ${relative_files})
        set(absolute_file "${dir}/${relative_file}")
        # determine link name for file including destination path
        set(symlink "${destination}/${relative_file}")

        # ensure that destination exists
        get_filename_component(symlink_dir "${symlink}" PATH)
        if(NOT EXISTS "${symlink_dir}")
          file(MAKE_DIRECTORY "${symlink_dir}")
        endif()

        _ament_cmake_symlink_install_create_symlink("${absolute_file}" "${symlink}")
      endforeach()
    else()
      if(NOT ARG_OPTIONAL)
        message(FATAL_ERROR
          "ament_cmake_symlink_install_directory() can't find '${dir}'")
      endif()
    endif()
  endforeach()
endfunction()

#
# Reimplement CMake install(FILES) command to use symlinks instead of copying
# resources.
#
# :param cmake_current_source_dir: The CMAKE_CURRENT_SOURCE_DIR when install
#   was invoked
# :type cmake_current_source_dir: string
# :param ARGN: the same arguments as the CMake install command.
# :type ARGN: various
#
function(ament_cmake_symlink_install_files cmake_current_source_dir)
  cmake_parse_arguments(ARG "OPTIONAL" "DESTINATION;RENAME" "FILES" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "ament_cmake_symlink_install_files() called with "
      "unused/unsupported arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  # make destination an absolute path and ensure that it exists
  if(NOT IS_ABSOLUTE "${ARG_DESTINATION}")
    set(ARG_DESTINATION "/driverless/driverless_ws/install/eufs_msgs/${ARG_DESTINATION}")
  endif()
  if(NOT EXISTS "${ARG_DESTINATION}")
    file(MAKE_DIRECTORY "${ARG_DESTINATION}")
  endif()

  if(ARG_RENAME)
    list(LENGTH ARG_FILES file_count)
    if(NOT file_count EQUAL 1)
    message(FATAL_ERROR "ament_cmake_symlink_install_files() called with "
      "RENAME argument but not with a single file")
    endif()
  endif()

  # iterate over files
  foreach(file ${ARG_FILES})
    # make file an absolute path
    if(NOT IS_ABSOLUTE "${file}")
      set(file "${cmake_current_source_dir}/${file}")
    endif()

    if(EXISTS "${file}")
      # determine link name for file including destination path
      get_filename_component(filename "${file}" NAME)
      if(NOT ARG_RENAME)
        set(symlink "${ARG_DESTINATION}/${filename}")
      else()
        set(symlink "${ARG_DESTINATION}/${ARG_RENAME}")
      endif()
      _ament_cmake_symlink_install_create_symlink("${file}" "${symlink}")
    else()
      if(NOT ARG_OPTIONAL)
        message(FATAL_ERROR
          "ament_cmake_symlink_install_files() can't find '${file}'")
      endif()
    endif()
  endforeach()
endfunction()

#
# Reimplement CMake install(PROGRAMS) command to use symlinks instead of copying
# resources.
#
# :param cmake_current_source_dir: The CMAKE_CURRENT_SOURCE_DIR when install
#   was invoked
# :type cmake_current_source_dir: string
# :param ARGN: the same arguments as the CMake install command.
# :type ARGN: various
#
function(ament_cmake_symlink_install_programs cmake_current_source_dir)
  cmake_parse_arguments(ARG "OPTIONAL" "DESTINATION" "PROGRAMS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "ament_cmake_symlink_install_programs() called with "
      "unused/unsupported arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  # make destination an absolute path and ensure that it exists
  if(NOT IS_ABSOLUTE "${ARG_DESTINATION}")
    set(ARG_DESTINATION "/driverless/driverless_ws/install/eufs_msgs/${ARG_DESTINATION}")
  endif()
  if(NOT EXISTS "${ARG_DESTINATION}")
    file(MAKE_DIRECTORY "${ARG_DESTINATION}")
  endif()

  # iterate over programs
  foreach(file ${ARG_PROGRAMS})
    # make file an absolute path
    if(NOT IS_ABSOLUTE "${file}")
      set(file "${cmake_current_source_dir}/${file}")
    endif()

    if(EXISTS "${file}")
      # determine link name for file including destination path
      get_filename_component(filename "${file}" NAME)
      set(symlink "${ARG_DESTINATION}/${filename}")
      _ament_cmake_symlink_install_create_symlink("${file}" "${symlink}")
    else()
      if(NOT ARG_OPTIONAL)
        message(FATAL_ERROR
          "ament_cmake_symlink_install_programs() can't find '${file}'")
      endif()
    endif()
  endforeach()
endfunction()

#
# Reimplement CMake install(TARGETS) command to use symlinks instead of copying
# resources.
#
# :param TARGET_FILES: the absolute files, replacing the name of targets passed
#   in as TARGETS
# :type TARGET_FILES: list of files
# :param ARGN: the same arguments as the CMake install command except that
#   keywords identifying the kind of type and the DESTINATION keyword must be
#   joined with an underscore, e.g. ARCHIVE_DESTINATION.
# :type ARGN: various
#
function(ament_cmake_symlink_install_targets)
  cmake_parse_arguments(ARG "OPTIONAL" "ARCHIVE_DESTINATION;DESTINATION;LIBRARY_DESTINATION;RUNTIME_DESTINATION"
    "TARGETS;TARGET_FILES" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "ament_cmake_symlink_install_targets() called with "
      "unused/unsupported arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  # iterate over target files
  foreach(file ${ARG_TARGET_FILES})
    if(NOT IS_ABSOLUTE "${file}")
      message(FATAL_ERROR "ament_cmake_symlink_install_targets() target file "
        "'${file}' must be an absolute path")
    endif()

    # determine destination of file based on extension
    set(destination "")
    get_filename_component(fileext "${file}" EXT)
    if(fileext STREQUAL ".a" OR fileext STREQUAL ".lib")
      set(destination "${ARG_ARCHIVE_DESTINATION}")
    elseif(fileext STREQUAL ".dylib" OR fileext MATCHES "\\.so(\\.[0-9]+)?(\\.[0-9]+)?(\\.[0-9]+)?$")
      set(destination "${ARG_LIBRARY_DESTINATION}")
    elseif(fileext STREQUAL "" OR fileext STREQUAL ".dll" OR fileext STREQUAL ".exe")
      set(destination "${ARG_RUNTIME_DESTINATION}")
    endif()
    if(destination STREQUAL "")
      set(destination "${ARG_DESTINATION}")
    endif()

    # make destination an absolute path and ensure that it exists
    if(NOT IS_ABSOLUTE "${destination}")
      set(destination "/driverless/driverless_ws/install/eufs_msgs/${destination}")
    endif()
    if(NOT EXISTS "${destination}")
      file(MAKE_DIRECTORY "${destination}")
    endif()

    if(EXISTS "${file}")
      # determine link name for file including destination path
      get_filename_component(filename "${file}" NAME)
      set(symlink "${destination}/${filename}")
      _ament_cmake_symlink_install_create_symlink("${file}" "${symlink}")
    else()
      if(NOT ARG_OPTIONAL)
        message(FATAL_ERROR
          "ament_cmake_symlink_install_targets() can't find '${file}'")
      endif()
    endif()
  endforeach()
endfunction()

function(_ament_cmake_symlink_install_create_symlink absolute_file symlink)
  # register symlink for being removed during install step
  file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/symlink_install_manifest.txt"
    "${symlink}\n")

  # avoid any work if correct symlink is already in place
  if(EXISTS "${symlink}" AND IS_SYMLINK "${symlink}")
    get_filename_component(destination "${symlink}" REALPATH)
    get_filename_component(real_absolute_file "${absolute_file}" REALPATH)
    if(destination STREQUAL real_absolute_file)
      message(STATUS "Up-to-date symlink: ${symlink}")
      return()
    endif()
  endif()

  message(STATUS "Symlinking: ${symlink}")
  if(EXISTS "${symlink}" OR IS_SYMLINK "${symlink}")
    file(REMOVE "${symlink}")
  endif()

  execute_process(
    COMMAND "/usr/bin/cmake" "-E" "create_symlink"
      "${absolute_file}"
      "${symlink}"
  )
  # the CMake command does not provide a return code so check manually
  if(NOT EXISTS "${symlink}" OR NOT IS_SYMLINK "${symlink}")
    get_filename_component(destination "${symlink}" REALPATH)
    message(FATAL_ERROR
      "Could not create symlink '${symlink}' pointing to '${absolute_file}'")
  endif()
endfunction()

# end of template

message(STATUS "Execute custom install script")

# begin of custom install code

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_index/share/ament_index/resource_index/rosidl_interfaces/eufs_msgs" "DESTINATION" "share/ament_index/resource_index/rosidl_interfaces")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_index/share/ament_index/resource_index/rosidl_interfaces/eufs_msgs" "DESTINATION" "share/ament_index/resource_index/rosidl_interfaces")

# install(DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_c/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN" "*.h")
ament_cmake_symlink_install_directory("/driverless/driverless_ws/src/eufs_msgs" DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_c/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN" "*.h")

# install(FILES "/opt/ros/foxy/lib/python3.8/site-packages/ament_package/template/environment_hook/library_path.sh" "DESTINATION" "share/eufs_msgs/environment")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/opt/ros/foxy/lib/python3.8/site-packages/ament_package/template/environment_hook/library_path.sh" "DESTINATION" "share/eufs_msgs/environment")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/library_path.dsv" "DESTINATION" "share/eufs_msgs/environment")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/library_path.dsv" "DESTINATION" "share/eufs_msgs/environment")

# install(DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_typesupport_fastrtps_c/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN_EXCLUDE" "*.cpp")
ament_cmake_symlink_install_directory("/driverless/driverless_ws/src/eufs_msgs" DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_typesupport_fastrtps_c/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN_EXCLUDE" "*.cpp")

# install("TARGETS" "eufs_msgs__rosidl_typesupport_fastrtps_c" "ARCHIVE_DESTINATION" "lib" "LIBRARY_DESTINATION" "lib" "RUNTIME_DESTINATION" "bin")
include("/driverless/driverless_ws/build/eufs_msgs/ament_cmake_symlink_install_targets_0_${CMAKE_INSTALL_CONFIG_NAME}.cmake")

# install(DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_typesupport_fastrtps_cpp/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN_EXCLUDE" "*.cpp")
ament_cmake_symlink_install_directory("/driverless/driverless_ws/src/eufs_msgs" DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_typesupport_fastrtps_cpp/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN_EXCLUDE" "*.cpp")

# install("TARGETS" "eufs_msgs__rosidl_typesupport_fastrtps_cpp" "ARCHIVE_DESTINATION" "lib" "LIBRARY_DESTINATION" "lib" "RUNTIME_DESTINATION" "bin")
include("/driverless/driverless_ws/build/eufs_msgs/ament_cmake_symlink_install_targets_1_${CMAKE_INSTALL_CONFIG_NAME}.cmake")

# install(DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_typesupport_introspection_c/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN" "*.h")
ament_cmake_symlink_install_directory("/driverless/driverless_ws/src/eufs_msgs" DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_typesupport_introspection_c/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN" "*.h")

# install(DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_cpp/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN" "*.hpp")
ament_cmake_symlink_install_directory("/driverless/driverless_ws/src/eufs_msgs" DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_cpp/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN" "*.hpp")

# install(DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_typesupport_introspection_cpp/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN" "*.hpp")
ament_cmake_symlink_install_directory("/driverless/driverless_ws/src/eufs_msgs" DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_typesupport_introspection_cpp/eufs_msgs/" "DESTINATION" "include/eufs_msgs" "PATTERN" "*.hpp")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/pythonpath.sh" "DESTINATION" "share/eufs_msgs/environment")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/pythonpath.sh" "DESTINATION" "share/eufs_msgs/environment")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/pythonpath.dsv" "DESTINATION" "share/eufs_msgs/environment")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/pythonpath.dsv" "DESTINATION" "share/eufs_msgs/environment")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_py/eufs_msgs/__init__.py" "DESTINATION" "lib/python3.8/site-packages/eufs_msgs")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_py/eufs_msgs/__init__.py" "DESTINATION" "lib/python3.8/site-packages/eufs_msgs")

# install(DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_py/eufs_msgs/msg/" "DESTINATION" "lib/python3.8/site-packages/eufs_msgs/msg" "PATTERN" "*.py")
ament_cmake_symlink_install_directory("/driverless/driverless_ws/src/eufs_msgs" DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_py/eufs_msgs/msg/" "DESTINATION" "lib/python3.8/site-packages/eufs_msgs/msg" "PATTERN" "*.py")

# install(DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_py/eufs_msgs/action/" "DESTINATION" "lib/python3.8/site-packages/eufs_msgs/action" "PATTERN" "*.py")
ament_cmake_symlink_install_directory("/driverless/driverless_ws/src/eufs_msgs" DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_py/eufs_msgs/action/" "DESTINATION" "lib/python3.8/site-packages/eufs_msgs/action" "PATTERN" "*.py")

# install(DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_py/eufs_msgs/srv/" "DESTINATION" "lib/python3.8/site-packages/eufs_msgs/srv" "PATTERN" "*.py")
ament_cmake_symlink_install_directory("/driverless/driverless_ws/src/eufs_msgs" DIRECTORY "/driverless/driverless_ws/build/eufs_msgs/rosidl_generator_py/eufs_msgs/srv/" "DESTINATION" "lib/python3.8/site-packages/eufs_msgs/srv" "PATTERN" "*.py")

# install("TARGETS" "eufs_msgs__rosidl_typesupport_fastrtps_c__pyext" "DESTINATION" "lib/python3.8/site-packages/eufs_msgs")
include("/driverless/driverless_ws/build/eufs_msgs/ament_cmake_symlink_install_targets_2_${CMAKE_INSTALL_CONFIG_NAME}.cmake")

# install("TARGETS" "eufs_msgs__rosidl_typesupport_introspection_c__pyext" "DESTINATION" "lib/python3.8/site-packages/eufs_msgs")
include("/driverless/driverless_ws/build/eufs_msgs/ament_cmake_symlink_install_targets_3_${CMAKE_INSTALL_CONFIG_NAME}.cmake")

# install("TARGETS" "eufs_msgs__rosidl_typesupport_c__pyext" "DESTINATION" "lib/python3.8/site-packages/eufs_msgs")
include("/driverless/driverless_ws/build/eufs_msgs/ament_cmake_symlink_install_targets_4_${CMAKE_INSTALL_CONFIG_NAME}.cmake")

# install("TARGETS" "eufs_msgs__python" "ARCHIVE_DESTINATION" "lib" "LIBRARY_DESTINATION" "lib" "RUNTIME_DESTINATION" "bin")
include("/driverless/driverless_ws/build/eufs_msgs/ament_cmake_symlink_install_targets_5_${CMAKE_INSTALL_CONFIG_NAME}.cmake")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/BoundingBox.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/BoundingBox.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/BoundingBoxes.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/BoundingBoxes.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/CanState.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/CanState.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/CarState.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/CarState.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/ChassisCommand.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/ChassisCommand.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/ChassisState.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/ChassisState.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/ConeArray.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/ConeArray.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/ConeArrayWithCovariance.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/ConeArrayWithCovariance.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/ConeWithCovariance.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/ConeWithCovariance.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/Costmap.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/Costmap.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/EKFErr.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/EKFErr.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/EKFState.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/EKFState.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/FullState.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/FullState.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/Heartbeat.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/Heartbeat.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/IntegrationErr.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/IntegrationErr.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/LapStats.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/LapStats.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/MPCState.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/MPCState.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/NodeState.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/NodeState.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/NodeStateArray.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/NodeStateArray.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PathIntegralParams.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PathIntegralParams.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PathIntegralStats.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PathIntegralStats.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PathIntegralStatus.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PathIntegralStatus.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PathIntegralTiming.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PathIntegralTiming.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PlanningMode.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PlanningMode.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PointArray.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PointArray.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PointArrayStamped.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PointArrayStamped.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PurePursuitCheckpoint.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PurePursuitCheckpoint.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PurePursuitCheckpointArrayStamped.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/PurePursuitCheckpointArrayStamped.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/Runstop.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/Runstop.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/SLAMErr.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/SLAMErr.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/SLAMState.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/SLAMState.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/StateMachineState.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/StateMachineState.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/SystemStatus.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/SystemStatus.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/TopicStatus.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/TopicStatus.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/VehicleCommands.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/VehicleCommands.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/VehicleCommandsStamped.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/VehicleCommandsStamped.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/Waypoint.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/Waypoint.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/WaypointArrayStamped.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/WaypointArrayStamped.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/WheelOdometryErr.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/WheelOdometryErr.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/WheelSpeeds.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/WheelSpeeds.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/WheelSpeedsStamped.idl" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/msg/WheelSpeedsStamped.idl" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/action/CheckForObjects.idl" "DESTINATION" "share/eufs_msgs/action")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/action/CheckForObjects.idl" "DESTINATION" "share/eufs_msgs/action")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/srv/Register.idl" "DESTINATION" "share/eufs_msgs/srv")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/srv/Register.idl" "DESTINATION" "share/eufs_msgs/srv")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/srv/SetCanState.idl" "DESTINATION" "share/eufs_msgs/srv")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_adapter/eufs_msgs/srv/SetCanState.idl" "DESTINATION" "share/eufs_msgs/srv")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/BoundingBox.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/BoundingBox.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/BoundingBoxes.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/BoundingBoxes.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/CanState.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/CanState.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/CarState.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/CarState.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/ChassisCommand.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/ChassisCommand.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/ChassisState.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/ChassisState.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/ConeArray.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/ConeArray.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/ConeArrayWithCovariance.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/ConeArrayWithCovariance.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/ConeWithCovariance.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/ConeWithCovariance.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/Costmap.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/Costmap.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/EKFErr.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/EKFErr.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/EKFState.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/EKFState.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/FullState.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/FullState.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/Heartbeat.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/Heartbeat.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/IntegrationErr.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/IntegrationErr.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/LapStats.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/LapStats.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/MPCState.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/MPCState.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/NodeState.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/NodeState.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/NodeStateArray.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/NodeStateArray.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PathIntegralParams.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PathIntegralParams.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PathIntegralStats.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PathIntegralStats.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PathIntegralStatus.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PathIntegralStatus.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PathIntegralTiming.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PathIntegralTiming.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PlanningMode.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PlanningMode.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PointArray.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PointArray.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PointArrayStamped.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PointArrayStamped.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PurePursuitCheckpoint.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PurePursuitCheckpoint.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PurePursuitCheckpointArrayStamped.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/PurePursuitCheckpointArrayStamped.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/Runstop.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/Runstop.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/SLAMErr.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/SLAMErr.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/SLAMState.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/SLAMState.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/StateMachineState.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/StateMachineState.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/SystemStatus.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/SystemStatus.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/TopicStatus.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/TopicStatus.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/VehicleCommands.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/VehicleCommands.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/VehicleCommandsStamped.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/VehicleCommandsStamped.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/Waypoint.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/Waypoint.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/WaypointArrayStamped.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/WaypointArrayStamped.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/WheelOdometryErr.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/WheelOdometryErr.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/WheelSpeeds.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/WheelSpeeds.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/msg/WheelSpeedsStamped.msg" "DESTINATION" "share/eufs_msgs/msg")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/msg/WheelSpeedsStamped.msg" "DESTINATION" "share/eufs_msgs/msg")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/action/CheckForObjects.action" "DESTINATION" "share/eufs_msgs/action")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/action/CheckForObjects.action" "DESTINATION" "share/eufs_msgs/action")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/srv/Register.srv" "DESTINATION" "share/eufs_msgs/srv")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/srv/Register.srv" "DESTINATION" "share/eufs_msgs/srv")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/srv/Register_Request.msg" "DESTINATION" "share/eufs_msgs/srv")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/srv/Register_Request.msg" "DESTINATION" "share/eufs_msgs/srv")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/srv/Register_Response.msg" "DESTINATION" "share/eufs_msgs/srv")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/srv/Register_Response.msg" "DESTINATION" "share/eufs_msgs/srv")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/srv/SetCanState.srv" "DESTINATION" "share/eufs_msgs/srv")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/srv/SetCanState.srv" "DESTINATION" "share/eufs_msgs/srv")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/srv/SetCanState_Request.msg" "DESTINATION" "share/eufs_msgs/srv")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/srv/SetCanState_Request.msg" "DESTINATION" "share/eufs_msgs/srv")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/srv/SetCanState_Response.msg" "DESTINATION" "share/eufs_msgs/srv")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/srv/SetCanState_Response.msg" "DESTINATION" "share/eufs_msgs/srv")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/eufs_msgs" "DESTINATION" "share/ament_index/resource_index/package_run_dependencies")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/eufs_msgs" "DESTINATION" "share/ament_index/resource_index/package_run_dependencies")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/eufs_msgs" "DESTINATION" "share/ament_index/resource_index/parent_prefix_path")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/eufs_msgs" "DESTINATION" "share/ament_index/resource_index/parent_prefix_path")

# install(FILES "/opt/ros/foxy/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh" "DESTINATION" "share/eufs_msgs/environment")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/opt/ros/foxy/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh" "DESTINATION" "share/eufs_msgs/environment")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/ament_prefix_path.dsv" "DESTINATION" "share/eufs_msgs/environment")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/ament_prefix_path.dsv" "DESTINATION" "share/eufs_msgs/environment")

# install(FILES "/opt/ros/foxy/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh" "DESTINATION" "share/eufs_msgs/environment")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/opt/ros/foxy/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh" "DESTINATION" "share/eufs_msgs/environment")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/path.dsv" "DESTINATION" "share/eufs_msgs/environment")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/path.dsv" "DESTINATION" "share/eufs_msgs/environment")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/local_setup.bash" "DESTINATION" "share/eufs_msgs")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/local_setup.bash" "DESTINATION" "share/eufs_msgs")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/local_setup.sh" "DESTINATION" "share/eufs_msgs")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/local_setup.sh" "DESTINATION" "share/eufs_msgs")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/local_setup.zsh" "DESTINATION" "share/eufs_msgs")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/local_setup.zsh" "DESTINATION" "share/eufs_msgs")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/local_setup.dsv" "DESTINATION" "share/eufs_msgs")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/local_setup.dsv" "DESTINATION" "share/eufs_msgs")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/package.dsv" "DESTINATION" "share/eufs_msgs")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_environment_hooks/package.dsv" "DESTINATION" "share/eufs_msgs")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_index/share/ament_index/resource_index/packages/eufs_msgs" "DESTINATION" "share/ament_index/resource_index/packages")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_index/share/ament_index/resource_index/packages/eufs_msgs" "DESTINATION" "share/ament_index/resource_index/packages")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/rosidl_cmake-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/rosidl_cmake-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_export_dependencies/ament_cmake_export_dependencies-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_export_dependencies/ament_cmake_export_dependencies-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_export_libraries/ament_cmake_export_libraries-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_export_libraries/ament_cmake_export_libraries-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_export_targets/ament_cmake_export_targets-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_export_targets/ament_cmake_export_targets-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_export_include_directories/ament_cmake_export_include_directories-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_export_include_directories/ament_cmake_export_include_directories-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/rosidl_cmake_export_typesupport_libraries-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/rosidl_cmake_export_typesupport_libraries-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/rosidl_cmake_export_typesupport_targets-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/rosidl_cmake/rosidl_cmake_export_typesupport_targets-extras.cmake" "DESTINATION" "share/eufs_msgs/cmake")

# install(FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_core/eufs_msgsConfig.cmake" "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_core/eufs_msgsConfig-version.cmake" "DESTINATION" "share/eufs_msgs/cmake")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_core/eufs_msgsConfig.cmake" "/driverless/driverless_ws/build/eufs_msgs/ament_cmake_core/eufs_msgsConfig-version.cmake" "DESTINATION" "share/eufs_msgs/cmake")

# install(FILES "/driverless/driverless_ws/src/eufs_msgs/package.xml" "DESTINATION" "share/eufs_msgs")
ament_cmake_symlink_install_files("/driverless/driverless_ws/src/eufs_msgs" FILES "/driverless/driverless_ws/src/eufs_msgs/package.xml" "DESTINATION" "share/eufs_msgs")
