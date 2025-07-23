setupHeaders(
        HEADERS
            MissionResponse.hpp
            MissionResponseExporter3540.hpp
            MissionResponseImporter3540.hpp

    GROUP datablocks\\missionresponse
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)