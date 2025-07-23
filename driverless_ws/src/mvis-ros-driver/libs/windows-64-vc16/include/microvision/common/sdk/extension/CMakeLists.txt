setupHeaders(
    HEADERS
        Extendable.hpp

        ExporterFactory.hpp
        ExporterFactoryExtension.hpp

        ImporterFactory.hpp
        ImporterFactoryExtension.hpp

    GROUP extension
    TARGET sdk-core
)

setupHeaders(
    HEADERS
        Extendable.hpp

        ConfigurationFactory.hpp
        ConfigurationFactoryExtension.hpp

        DeviceFactory.hpp
        DeviceFactoryExtension.hpp

        StreamReaderFactory.hpp
        StreamReaderFactoryExtension.hpp
        
        StreamWriterFactory.hpp
        StreamWriterFactoryExtension.hpp

        NetworkInterfaceFactory.hpp
        NetworkInterfaceFactoryExtension.hpp

    GROUP extension
    TARGET sdk-devices
)

include(${CMAKE_CURRENT_LIST_DIR}/io/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/config/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/devices/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/datablocks/CMakeLists.txt)
