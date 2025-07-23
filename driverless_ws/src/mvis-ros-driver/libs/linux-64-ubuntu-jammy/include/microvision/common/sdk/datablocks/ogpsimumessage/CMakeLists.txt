setupHeaders(
    HEADERS
        OGpsImuMessage.hpp
        OGpsImuMessageExporter2610.hpp
        OGpsImuMessageImporter2610.hpp

    GROUP datablocks\\OGpsImuMessage
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
