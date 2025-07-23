//==============================================================================
//! \file
//!
//! \brief Definition of device accessing image stream of RTSP.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 20, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/devices/RtspImageStreamConfiguration.hpp>

#include <microvision/common/sdk/datablocks/ImporterBase.hpp>
#include <microvision/common/sdk/devices/IdcDevice.hpp>
#include <microvision/common/sdk/misc/BackgroundWorker.hpp>
#include <microvision/common/sdk/extension/NetworkInterfaceFactory.hpp>

#include <microvision/common/sdk/io/http/HttpRequestNetworkDataPackage.hpp>
#include <microvision/common/sdk/io/http/HttpResponseNetworkDataPackage.hpp>
#include <microvision/common/sdk/io/http/HttpToNetworkPackageTranslator.hpp>
#include <microvision/common/sdk/io/http/NetworkToHttpPackageTranslator.hpp>
#include <microvision/common/sdk/io/rtp/NetworkDataPackageToImageTranslator.hpp>

#include <microvision/common/logging/logging.hpp>

#include <map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Access image stream of RTSP device.
//! \extends IdcDevice
//------------------------------------------------------------------------------
class RtspImageStreamDevice final : public IdcDevice
{
public:
    //========================================
    //! \brief Clock type to be used for timed operations.
    //----------------------------------------
    using ClockType = std::chrono::high_resolution_clock;

public:
    //========================================
    //! \brief Type name of device.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

    //========================================
    //! \brief Type name of device.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string& getTypeName();

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::RtspImageStreamDevice";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    RtspImageStreamDevice();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~RtspImageStreamDevice() override;

public: // implements Configurable
    //========================================
    //! \brief Get supported types of configuration.
    //!
    //! Supported means that this device can interpret imported data packages using this configuration.
    //! Configuration type is a human readable unique string name of the configuration
    //! used to address it in code.
    //!
    //! \returns All supported configuration types.
    //----------------------------------------
    const std::vector<std::string>& getConfigurationTypes() const override;

public: // implements Device
    //========================================
    //! \brief Get the type name of this device.
    //! \returns The type name of this device.
    //----------------------------------------
    const std::string& getDeviceType() const override;

    //========================================
    //! \brief Get the last error which has occurred.
    //! \returns The exception of the last occurred error.
    //----------------------------------------
    std::exception_ptr getLastError() const override;

    //========================================
    //! \brief Checks if a connection has been established.
    //! \note This reacts in an asynchronous manner to the connect() method in await of the handshake.
    //! \returns Either \c true if connection is established or otherwise \c false.
    //----------------------------------------
    bool isConnected() const override;

    //========================================
    //! \brief Checks that a connection is established and packages are still processed.
    //! \returns Either \c true if connection is established or work is ongoing on received data or otherwise \c false.
    //----------------------------------------
    bool isWorking() override;

    //========================================
    //! \brief Connects the device to the physical device via network.
    //----------------------------------------
    void connect() override;

    //========================================
    //! \brief Disconnects the device from the physical device.
    //----------------------------------------
    void disconnect() override;

    //========================================
    //! \brief Get the current configuration of the device.
    //! \returns The current configuration.
    //----------------------------------------
    ConfigurationPtr getDeviceConfiguration() const override;

    //========================================
    //! \brief Set the configuration of this device.
    //! \param[in] deviceConfiguration  The shared pointer to new configuration of this device.
    //! \return Either \c true if this device can be configured with those configuration, otherwise \c false.
    //----------------------------------------
    bool setDeviceConfiguration(const ConfigurationPtr& deviceConfiguration) override;

    //========================================
    //! \brief Lock current configuration for use.
    //!
    //! The locked configuration will be used for all received data packages until unlock.
    //! Configuration will be locked during connection - some properties may still be updated when set
    //! depending on device implementation.
    //! For offline use with processDataPackage the configuration can be manually unlocked to change
    //! configuration properties. Then before the next processDataPackage it has to be locked again.
    //!
    //! \return Either \c true if the configuration is valid, otherwise \c false.
    //----------------------------------------
    bool lockConfiguration() override;

    //========================================
    //! \brief Release locked configuration.
    //!
    //! This does not work while the device is connected.
    //!
    //! \return Either \c true if the configuration is unlocked, otherwise \c false.
    //----------------------------------------
    bool unlockConfiguration() override;

    //========================================
    //! \brief Process a data package as sensor input from another source (e.g. as read from a file).
    //! \param[in] dataPackage  The data package to process.
    //! \return Either \c true if the processing was successful, otherwise \c false.
    //----------------------------------------
    bool processDataPackage(const DataPackagePtr& dataPackage) override;

private:
    //========================================
    //! \brief Main method of thread which observes the receiver.
    //! \param[in] worker  Executing Background worker m_observerWorker.
    //! \returns Either \c true too keep thread alive or otherwise \c false.
    //----------------------------------------
    bool observerMain(BackgroundWorker& worker);

    //========================================
    //! \brief Process received RTSP command response.
    //! \param[in] package  RTSP command response as HTTP data package.
    //----------------------------------------
    void processRtspCommandResponse(const HttpNetworkDataPackagePtr& package);

    //========================================
    //! \brief Process received RTSP describe command response.
    //! \param[in] package  RTSP command response as HTTP data package.
    //----------------------------------------
    void processRtspDescribeCommandResponse(const HttpNetworkDataPackagePtr& package);

    //========================================
    //! \brief Process received RTSP setup command response.
    //! \param[in] package  RTSP command response as HTTP data package.
    //----------------------------------------
    void processRtspSetupCommandResponse(const HttpNetworkDataPackagePtr& package);

    //========================================
    //! \brief Publish idc package as received by device to listeners.
    //! \param[in] data  Shared idc data package pointer of received data.
    //----------------------------------------
    void publishIdcPackage(const IdcDataPackagePtr& data);

    //========================================
    //! \brief Method to be called if a new IdcDataPackage has been received.
    //! \param[in] data  Shared idc data package pointer of received data.
    //----------------------------------------
    void notifyListeners(const IdcDataPackagePtr& data);

private:
    //========================================
    //! \brief Increment index of received packages.
    //----------------------------------------
    ThreadSafe<int64_t> m_packageIndex;

    //========================================
    //! \brief The Uri which denotes the packets dedicated for this device.
    //----------------------------------------
    Uri m_packageUri;

    //========================================
    //! \brief HTTP url to address RTSP service.
    //----------------------------------------
    std::string m_rtspServiceUrl;

    //========================================
    //! \brief RTSP image stream session id.
    //----------------------------------------
    std::string m_rtspSession;

    //========================================
    //! \brief Receiver observer thread.
    //----------------------------------------
    BackgroundWorker m_observerWorker;

    //========================================
    //! \brief Handle to wait for all network resources on input.
    //----------------------------------------
    ThreadSyncPtr m_networkResourceSyncHandle;

    //========================================
    //! \brief Timestamp to trigger keep alive message on RTSP connection.
    //----------------------------------------
    ClockType::time_point m_nextMessageToKeepRtspConnectionAlive;

    //========================================
    //! \brief Disconnect timeout to avoid blockage.
    //----------------------------------------
    ClockType::time_point m_disconnectTimeout;

    //========================================
    //! \brief TCP client for RTSP communication.
    //!
    //! Over the RTSP protocol the device can be controlled.
    //! To request an image stream for example.
    //----------------------------------------
    NetworkInterfaceUPtr m_tcpClientForRtspStreamControl;

    //========================================
    //! \brief UDP Receiver for RTP data packages.
    //!
    //! Via the RTP protocol the image data will be received.
    //----------------------------------------
    NetworkInterfaceUPtr m_udpReceiverForRtpImageStream;

    //========================================
    //! \brief UDP Receiver for RTCP data packages.
    //!
    //! Via the RTCP protocol the stream quality reports will be received.
    //----------------------------------------
    NetworkInterfaceUPtr m_udpReceiverForRtcpTimeSync;

    //========================================
    //! \brief UDP Sender for RTP dummy messages
    //----------------------------------------
    NetworkInterfaceUPtr m_udpSenderForRtpDummyMessages;

    //========================================
    //! \brief UDP Sender for RTCP dummy mesages.
    //----------------------------------------
    NetworkInterfaceUPtr m_udpSenderForRtcpDummyMessages;

    //========================================
    //! \brief Current RTSP command which has been sent.
    //----------------------------------------
    HttpRequestNetworkDataPackagePtr m_currentRtspRequest;

    //========================================
    //! \brief Translates HTTP to network data packages and transmit it by RSTP connection.
    //----------------------------------------
    HttpToNetworkPackageTranslator m_rtspCommandToSendTranslator;

    //========================================
    //! \brief Translates network to HTTP data packages and process by \a processRtspCommandResponse(...).
    //----------------------------------------
    NetworkToHttpPackageTranslator m_rtspCommandToReceiveTranslator;

    //========================================
    //! \brief Translates network to idc data packages of type Image2403 to publish by \a publishIdcPackage(...).
    //----------------------------------------
    NetworkDataPackageToImageTranslator m_rtpAndRtcpDataToImageTranslator;

    //========================================
    //! \brief Current configuration.
    //----------------------------------------
    RtspImageStreamConfigurationPtr m_configuration;

    //========================================
    //! \brief Locked udp configuration.
    //----------------------------------------
    RtspImageStreamConfigurationPtr m_lockedConfiguration;

}; // class RtspImageStreamDevice

//==============================================================================
//! \brief Pointer to instance of \a RtspImageStreamDevice.
//------------------------------------------------------------------------------
using RtspStreamDevicePtr = std::unique_ptr<RtspImageStreamDevice>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
