setupHeaders(
    HEADERS
        Configurable.hpp
        Configuration.hpp
        ConfigurationProperty.hpp
        ConfigurationPropertyOf.hpp
        ConfigurationPropertyOfType.hpp
        EnumConfigurationProperty.hpp
        EnumConfigurationPropertyOf.hpp

    GROUP config
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/devices/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/importers/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/io/CMakeLists.txt)
