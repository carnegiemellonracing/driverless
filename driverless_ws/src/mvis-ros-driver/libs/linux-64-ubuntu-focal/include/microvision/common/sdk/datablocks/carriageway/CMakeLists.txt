setupHeaders(
    HEADERS
        CarriageWay.hpp
        CarriageWayList.hpp
        CarriageWayListExporter6972.hpp
        CarriageWayListImporter6970.hpp
        CarriageWayListImporter6972.hpp
        CarriageWayListSerializedSize6970.hpp
        CarriageWayListSerializedSize6972.hpp
        CarriageWaySegment.hpp
        Lane.hpp
        LaneSegment.hpp
        LaneSupportPoint.hpp

    GROUP datablocks\\carriageway
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
