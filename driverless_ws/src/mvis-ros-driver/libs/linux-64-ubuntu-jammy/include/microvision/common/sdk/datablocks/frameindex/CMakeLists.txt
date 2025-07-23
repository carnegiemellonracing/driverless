setupHeaders(
    HEADERS
        FrameIndex.hpp
        FrameIndexExporter6130.hpp
        FrameIndexImporter6130.hpp

    GROUP datablocks\\frameindex
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
