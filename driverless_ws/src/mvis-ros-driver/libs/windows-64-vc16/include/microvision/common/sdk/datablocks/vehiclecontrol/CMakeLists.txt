setupHeaders(
    HEADERS
        VehicleControl.hpp
        VehicleControlExporter9100.hpp
        VehicleControlImporter9100.hpp

    GROUP datablocks\\vehiclecontrol
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
