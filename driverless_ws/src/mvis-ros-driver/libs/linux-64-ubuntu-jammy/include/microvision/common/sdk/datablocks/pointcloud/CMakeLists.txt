setupHeaders(
    HEADERS
        PointBase.hpp
        PointCloud.hpp
        PointCloudBase.hpp
        PointCloudExporter7510.hpp
        PointCloudExporter7511.hpp
        PointCloudImporter7510.hpp
        PointCloudImporter7511.hpp
        PointCloudPoint.hpp
        ReferencePlane.hpp

    GROUP datablocks\\pointcloud
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
