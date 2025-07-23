# Verbose messages available with cmake 3.15.7, uncomment these if you use this version or greater.

include(CMakeFindDependencyMacro)
include(${CMAKE_CURRENT_LIST_DIR}/commonDependencies.cmake)
# include("${CMAKE_CURRENT_LIST_DIR}/commonTargets.cmake")

file(GLOB_RECURSE CMAKE_SCRIPTS ${CMAKE_CURRENT_LIST_DIR}/cmake-scripts/*.cmake)

foreach(CMAKE_SCRIPT ${CMAKE_SCRIPTS})
    include("${CMAKE_SCRIPT}")
endforeach()

