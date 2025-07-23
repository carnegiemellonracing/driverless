setupHeaders(
    HEADERS
        EventTag.hpp
        EventTagExporter7000.hpp
        EventTagImporter7000.hpp

    GROUP datablocks\\eventtag
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
