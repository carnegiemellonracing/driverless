setupHeaders(
    HEADERS
        TimeRelation.hpp
        TimeRelationsList.hpp
        TimeRelationsListImporter9011.hpp
        TimeRelationsListImporter9010.hpp
        TimeRelationsListExporter9011.hpp

    GROUP datablocks\\timerelation
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
