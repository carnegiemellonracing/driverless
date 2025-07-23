setupHeaders(
    HEADERS
        MavinDataFormat.hpp

    GROUP datablocks\\mavinRaw
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
