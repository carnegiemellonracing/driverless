setupHeaders(
    HEADERS
        IdcTrailer.hpp
        IdcTrailerExporter6120.hpp
        IdcTrailerImporter6120.hpp

    GROUP datablocks\\idctrailer
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
