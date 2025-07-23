setupHeaders(
    HEADERS
        VehicleState.hpp
        VehicleStateImporter2805.hpp
        VehicleStateImporter2806.hpp
        VehicleStateImporter2807.hpp
        VehicleStateImporter2808.hpp
        VehicleStateImporter2809.hpp
        VehicleStateExporter2805.hpp
        VehicleStateExporter2808.hpp
        VehicleStateExporter2809.hpp

    GROUP datablocks\\vehiclestate
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
