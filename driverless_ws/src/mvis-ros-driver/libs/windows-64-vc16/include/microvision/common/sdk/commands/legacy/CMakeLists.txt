setupHeaders(
    HEADERS
        LegacyCommandRequestResponseBase.hpp

    TARGET sdk-devices
)

include(${CMAKE_CURRENT_LIST_DIR}/ecu/CMakeLists.txt)
