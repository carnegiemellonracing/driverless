setupHeaders(
    HEADERS
        DeviceStatus.hpp
        DeviceStatusExporter6301.hpp
        DeviceStatusExporter6303.hpp
        DeviceStatusImporter6301.hpp
        DeviceStatusImporter6303.hpp
        DeviceStatusSerializedSize6301.hpp
        DeviceStatusSerializedSize6303.hpp
        SerialNumber.hpp
        Version448.hpp

    GROUP datablocks\\devicestatus
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
