setupHeaders(
    HEADERS
        ObjectAssociation.hpp
        ObjectAssociationList.hpp
        ObjectAssociationListExporter4001.hpp
        ObjectAssociationListImporter4001.hpp

    GROUP datablocks\\objectassociationlist
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
