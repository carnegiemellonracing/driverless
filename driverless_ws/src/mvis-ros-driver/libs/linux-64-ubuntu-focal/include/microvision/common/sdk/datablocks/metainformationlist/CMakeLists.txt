setupHeaders(
    HEADERS
        MetaInformationAppBaseConfig.hpp
        MetaInformationBase.hpp
        MetaInformationEcuId.hpp
        MetaInformationFactory.hpp
        MetaInformationKeywords.hpp
        MetaInformationList.hpp
        MetaInformationListExporter7110.hpp
        MetaInformationListImporter7110.hpp
        MetaInformationProcessingDescription.hpp
        MetaInformationUnsupported.hpp
        MetaInformationVersionNumber.hpp
        MetaInformationSyncBoxConfig.hpp
        MetaInformationConfigFile.hpp
        MetaInformationTripMetaData.hpp
    GROUP datablocks\\metainformationlist
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/special/CMakeLists.txt)
