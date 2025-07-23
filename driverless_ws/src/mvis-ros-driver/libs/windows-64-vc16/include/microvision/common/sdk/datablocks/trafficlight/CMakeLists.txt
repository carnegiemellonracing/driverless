setupHeaders(
    HEADERS
        TrafficLightState.hpp
        TrafficLightStateList.hpp
        TrafficLightStateListExporter3600.hpp
        TrafficLightStateListImporter3600.hpp

    GROUP datablocks\\trafficlight
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
