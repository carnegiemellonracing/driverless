setupHeaders(
    HEADERS
        ContentSeparator.hpp
        ContentSeparatorExporter7100.hpp
        ContentSeparatorImporter7100.hpp

    GROUP datablocks\\contentseparator
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)