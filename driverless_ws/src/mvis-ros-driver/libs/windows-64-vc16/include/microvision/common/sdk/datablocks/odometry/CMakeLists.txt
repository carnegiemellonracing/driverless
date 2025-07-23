setupHeaders(
    HEADERS
        Odometry.hpp
        OdometryExporter9002.hpp
        OdometryImporter9002.hpp
        OdometryExporter9003.hpp
        OdometryImporter9003.hpp

    GROUP datablocks\\odometry
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)