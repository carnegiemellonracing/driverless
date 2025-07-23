setupHeaders(
    HEADERS
        LaneMarking.hpp
        LaneMarkingList.hpp
        LaneMarkingListExporter6901.hpp
        LaneMarkingListImporter6901.hpp

    GROUP datablocks\\lanemarking
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
