setupHeaders(
    HEADERS
        GpsImu.hpp
        GpsImuExporter9001.hpp
        GpsImuExporter9004.hpp
        GpsImuImporter9001.hpp
        GpsImuImporter9004.hpp

    GROUP datablocks\\gpsimu
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)