setupHeaders(
    HEADERS
       Command.hpp
       CommandInterface.hpp
       CommandInterfaceTag.hpp
    TARGET
        sdk-devices
)

include(${CMAKE_CURRENT_LIST_DIR}/legacy/CMakeLists.txt)
