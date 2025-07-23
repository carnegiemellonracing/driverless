setupHeaders(
    HEADERS
        StateOfOperation.hpp
        StateOfOperationExporter9111.hpp
        StateOfOperationImporter9111.hpp

    GROUP datablocks\\stateofoperation
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
