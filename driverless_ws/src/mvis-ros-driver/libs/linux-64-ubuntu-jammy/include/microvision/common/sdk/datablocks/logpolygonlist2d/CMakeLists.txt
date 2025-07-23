setupHeaders(
    HEADERS
        LogPolygon2d.hpp
        LogPolygonList2d.hpp
        LogPolygonList2dExporter6817.hpp
        LogPolygonList2dImporter6817.hpp
        Polygon2d.hpp

    GROUP datablocks\\logpolygonlist2d
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
