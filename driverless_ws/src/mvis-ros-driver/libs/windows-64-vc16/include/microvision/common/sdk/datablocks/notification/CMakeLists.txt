setupHeaders(
    HEADERS
        Notification.hpp
        NotificationExporter2030.hpp
        NotificationImporter2030.hpp

    GROUP datablocks\\notification
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
