# Custom FindTinyXML2.cmake

set(TinyXML2_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../libs/windows-64-vc16")
set(TinyXML2_INCLUDE_DIRS "${TinyXML2_ROOT_DIR}/include/tinyxml2")
set(TinyXML2_LIBRARY "${TinyXML2_ROOT_DIR}/lib/tinyxml2.lib")

# Set the CMake module path
set(TinyXML2_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../libs/windows-64-vc16/share/tinyxml2")

# Include the TinyXML2 CMake configuration if it exists
if(EXISTS "${TinyXML2_CMAKE_DIR}/tinyxml2-config.cmake")
    include("${TinyXML2_CMAKE_DIR}/tinyxml2-config.cmake")
endif()

# If the imported target doesn't exist, create it
if(NOT TARGET tinyxml2::tinyxml2)
    add_library(tinyxml2::tinyxml2 UNKNOWN IMPORTED)
    set_target_properties(tinyxml2::tinyxml2 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${TinyXML2_INCLUDE_DIRS}"
    )
endif()

# Set the found flag
set(TinyXML2_FOUND TRUE)
set(tinyxml2_FOUND TRUE)

# Mark as advanced
mark_as_advanced(TinyXML2_INCLUDE_DIRS TinyXML2_LIBRARY)

# Print debug information
message(STATUS "TinyXML2_INCLUDE_DIRS: ${TinyXML2_INCLUDE_DIRS}")
message(STATUS "TinyXML2_LIBRARY: ${TinyXML2_LIBRARY}")
