setupHeaders(
    HEADERS
        Commander.hpp
        IdcDeviceBase.hpp
        IdcEthDevice.hpp
        IdcEthType.hpp
        IdcTcpDevice.hpp
        IdcTcpIpAcceptorBase.hpp
        IdcTcpIpAcceptorEcu.hpp
        IdcTcpIpAcceptorScala.hpp
        MeasurementTimeProvider.hpp
        TcpReceiveThreadEnsemble.hpp
        ThreadEnsemble.hpp
        UdpDemultiplexer.hpp
        UdpReceiveThreadEnsemble.hpp

    GROUP devices
    TARGET sdk-core
)

setupHeaders(
    HEADERS
        MvisEcuDevice.hpp
        MvisLdeDevice.hpp
        MvisEcuUdpDevice.hpp
        MvisLuxHr.hpp
        MvisLuxHrSensorDevice.hpp
        MvisLux.hpp
        MvisLuxSensorDevice.hpp
        MvisMiniLux.hpp
        MvisMiniLuxSensorDevice.hpp
        MicroVisionMoviaSensorDevice.hpp
        Scala.hpp
        ScalaSensorDevice.hpp
        MvisSyncBoxDevice.hpp
        MvisTrackingBox.hpp
        MvisTrackingBoxDevice.hpp
        RtspImageStreamDevice.hpp
        MicroVisionMavinSensorDevice.hpp

    GROUP devices
    TARGET sdk-devices
)

include(${CMAKE_CURRENT_LIST_DIR}/sutp/CMakeLists.txt)
