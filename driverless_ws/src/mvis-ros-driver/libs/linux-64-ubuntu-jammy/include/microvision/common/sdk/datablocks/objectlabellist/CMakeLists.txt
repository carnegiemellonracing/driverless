setupHeaders(
    HEADERS
        ObjectLabel.hpp
        ObjectLabelList.hpp
        ObjectLabelListExporter6503.hpp
        ObjectLabelListImporter6503.hpp

    GROUP datablocks\\objectlabellist
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
