# ##############################################################################
# Function to collect targets within dependency targets.
# Arguments:
#     TARGTS - List of targets
# ------------------------------------------------------------------------------
function(listTargetsRecursive)
    set(multiValueArgs TARGETS)
    cmake_parse_arguments(COLLECT_TARGETS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(_TARGETS)

    message(DEBUG "COLLECT_TARGETS IN: ${COLLECT_TARGETS_TARGETS}")

    foreach(target ${COLLECT_TARGETS_TARGETS})
    list(FIND _TARGETS "${target}" _TARGET_INDEX)

        if (${_TARGET_INDEX} EQUAL -1)
            if(TARGET ${target})
                get_target_property(_LINK_LIBS ${target} INTERFACE_LINK_LIBRARIES)

                if (_LINK_LIBS)
                    listTargetsRecursive(TARGETS ${_LINK_LIBS})
                    list(APPEND _TARGETS ${COLLECTED_TARGETS})
                endif()
            endif()

            list(APPEND _TARGETS ${target})
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _TARGETS)
    message(DEBUG "COLLECT_TARGETS OUT: ${_TARGETS}")
    set(COLLECTED_TARGETS ${_TARGETS} PARENT_SCOPE)
endfunction(listTargetsRecursive)

# ##############################################################################
# Function to collect binary locations from targets within dependency targets.
# Arguments:
#     TARGTS - List of targets
# ------------------------------------------------------------------------------
function(listTargetFileDirsRecursive)
    set(multiValueArgs TARGETS)
    cmake_parse_arguments(COLLECT_TARGET_BINARY_LOCATIONS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(_TARGET_BINARY_LOCATIONS)

    listTargetsRecursive(TARGETS ${COLLECT_TARGET_BINARY_LOCATIONS_TARGETS})

    message(DEBUG "COLLECTED_TARGETS: ${COLLECTED_TARGETS}")

    foreach(target ${COLLECTED_TARGETS})
        if(TARGET ${target})
            get_target_property(_TYPE ${target} TYPE)

            if (_TYPE AND NOT _TYPE STREQUAL "INTERFACE_LIBRARY")
                list(APPEND _TARGET_BINARY_LOCATIONS $<TARGET_FILE_DIR:${target}>)
            endif()
        elseif(EXISTS ${target})
            get_filename_component(_DIR_LOCATION ${target} PATH)
            list(APPEND _TARGET_BINARY_LOCATIONS ${_DIR_LOCATION})
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _TARGET_BINARY_LOCATIONS)
    message(DEBUG "TARGTE LOCATIONS: ${_TARGET_BINARY_LOCATIONS}")
    set(COLLECTED_TARGET_BINARY_LOCATIONS ${_TARGET_BINARY_LOCATIONS} PARENT_SCOPE)
endfunction(listTargetFileDirsRecursive)
