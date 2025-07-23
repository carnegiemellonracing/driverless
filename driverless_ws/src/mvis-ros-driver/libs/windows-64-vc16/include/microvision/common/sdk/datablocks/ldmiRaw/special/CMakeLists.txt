setupHeaders(
    HEADERS
        LdmiRawFrame2352.hpp
        LdmiRawFrame2352Exporter2352.hpp
        LdmiRawFrame2352Importer2352.hpp
        LdmiRawFrameFooterIn2352.hpp
        LdmiRawFrameHeaderIn2352.hpp
        LdmiRawFrameRowIn2352.hpp
        LdmiRawStaticInfoIn2352.hpp

        LdmiRawFrame2353.hpp
        LdmiRawFrame2353Exporter2353.hpp
        LdmiRawFrame2353Importer2353.hpp
        LdmiRawFrameFooterIn2353.hpp
        LdmiRawFrameHeaderIn2353.hpp
        LdmiRawFrameRowIn2353.hpp
        LdmiRawStaticInfoIn2353.hpp

        LdmiRawFrame2354.hpp
        LdmiRawFrame2354Exporter2354.hpp
        LdmiRawFrame2354Importer2354.hpp
        LdmiRawFrameFooterIn2354.hpp
        LdmiRawFrameHeaderIn2354.hpp
        LdmiRawFrameRowIn2354.hpp
        LdmiRawStaticInfoIn2354.hpp

    GROUP datablocks\\ldmiRaw
    TARGET sdk-core
)

setupHeaders(
    HEADERS
        IdcToIntpPackageOfLdmiRawFrame2352Translator.hpp
        IntpToIdcPackageOfLdmiRawFrame2352Translator.hpp
        IdcToIntpPackageOfLdmiRawFrame2353Translator.hpp
        IntpToIdcPackageOfLdmiRawFrame2353Translator.hpp
        IdcToIntpPackageOfLdmiRawFrame2354Translator.hpp
        IntpToIdcPackageOfLdmiRawFrame2354Translator.hpp

    GROUP datablocks\\ldmiRaw
    TARGET sdk-core
)
