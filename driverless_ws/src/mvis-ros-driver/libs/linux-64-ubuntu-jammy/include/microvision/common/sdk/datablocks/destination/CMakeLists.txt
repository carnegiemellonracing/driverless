setupHeaders(
    HEADERS
        Destination.hpp
        DestinationExporter3520.hpp
        DestinationExporter3521.hpp
        DestinationImporter3520.hpp
        DestinationImporter3521.hpp

    GROUP datablocks\\destination
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
