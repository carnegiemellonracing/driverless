setupHeaders(
    HEADERS
        VehicleRequest.hpp
        VehicleRequestExporter9105.hpp
        VehicleRequestImporter9105.hpp

    GROUP datablocks\\vehiclerequest
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
