setupHeaders(
    HEADERS
        TimeRecord.hpp
        TimeRecordExporter9000.hpp
        TimeRecordImporter9000.hpp

    GROUP datablocks\\timerecord
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)