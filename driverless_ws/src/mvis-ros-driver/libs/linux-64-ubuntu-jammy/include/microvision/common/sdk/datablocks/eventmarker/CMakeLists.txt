setupHeaders(
    HEADERS
        EventMarker.hpp
        EventMarkerExporter7001.hpp
        EventMarkerImporter7001.hpp

    GROUP datablocks\\EventMarker
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)