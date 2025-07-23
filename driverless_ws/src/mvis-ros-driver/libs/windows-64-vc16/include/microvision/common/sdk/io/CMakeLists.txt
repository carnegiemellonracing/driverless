setupHeaders(
    HEADERS
        DataPackage.hpp
        DataPackageNetworkReceiver.hpp
        DataPackageNetworkSender.hpp
        DataPackageStreamReader.hpp
        DataPackageStreamWriter.hpp
        DataStreamTranslator.hpp
        IdcDataPackage.hpp
        IdcDataPackageStreamReader.hpp
        IdcDataPackageStreamWriter.hpp
        IdcStreamBase.hpp
        NetworkBase.hpp
        NetworkDataPackage.hpp
        StreamBase.hpp

    GROUP io
    TARGET sdk-core
)

include(${CMAKE_CURRENT_LIST_DIR}/bag/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/http/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/icd/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/idc/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/intp/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/ip/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/iutp/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/rtsp/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/tcp/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/rtp/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/someip/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/udp/CMakeLists.txt)