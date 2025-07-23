//==============================================================================
//! \file
//!
//! \brief MVIS ECU UDP device class definition.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 18, 2023
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/devices/MvisEcuUdpConfiguration.hpp>
#include <microvision/common/sdk/datablocks/PerceptionPerformanceInfo.hpp>
#include <microvision/common/sdk/devices/IdcDevice.hpp>
#include <microvision/common/sdk/extension/NetworkInterfaceFactory.hpp>
#include <microvision/common/sdk/misc/BackgroundWorker.hpp>
#include <microvision/common/sdk/io/iutp/NetworkDataPackageToIutpTranslator.hpp>
#include <microvision/common/sdk/io/icd/IutpToIcdPackageTranslator.hpp>
#include <microvision/common/sdk/io/icd/IcdToIdcPackageTranslator.hpp>
#include <microvision/common/logging/logging.hpp>

#include <map>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Access MVIS ECU UDP device.
//!
//! Device to connect the MVIS ECU UDP which is an ECU to gether lidar sensor data via UDP to receive IDC data.
//!
//! The configuration of the ECU UDP device is basically the default UDP configuration
//! which receives only data from one network source.
//!
//! \extends IdcDevice
//------------------------------------------------------------------------------
class MvisEcuUdpDevice final : public IdcDevice
{
public:
    //========================================
    //! \brief Type name of MVIS ECU UDP device.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

    //========================================
    //! \brief IUTP stream id
    //----------------------------------------
    static constexpr uint8_t iutpStreamId{0x80};

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::MvisEcuUdpDevice";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    MvisEcuUdpDevice();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~MvisEcuUdpDevice() override;

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
    //!
    //! \sa MvisEcuUdpConfiguration
    //----------------------------------------
    ConfigurationPtr getDeviceConfiguration() const override;

    //========================================
    //! \brief Set the configuration of this device.
    //! \param[in] deviceConfiguration  The shared pointer to new configuration of this device.
    //! \return Either \c true if this device can be configured with those configuration, otherwise \c false.
    //!
    //! \sa MvisEcuUdpConfiguration
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
    //! \brief Released locked configuration.
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
    //! \return Either \c true too keep thread alive or otherwise \c false.
    //----------------------------------------
    bool observerMain(BackgroundWorker& worker);

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
    //! \brief Previous package size of received packages.
    //----------------------------------------
    ThreadSafe<uint32_t> m_previousPackageSize;

    //========================================
    //! \brief The Uri which denotes the packets dedicated for this device.
    //----------------------------------------
    Uri m_packageUri;

    //========================================
    //! \brief Receiver observer thread.
    //----------------------------------------
    BackgroundWorker m_observerWorker;

    //========================================
    //! \brief Receiver for UDP data packages.
    //----------------------------------------
    NetworkInterfaceUPtr m_udpReceiver;

    //========================================
    //! \brief Network to iutp data package translator.
    //!
    //! For all successfully translated packages the icd translator is called.
    //----------------------------------------
    NetworkDataPackageToIutpTranslator m_iutpPackageForIcdTranslator;

    //========================================
    //! \brief Iutp to icd data package translator.
    //!
    //! For all successfully translated packages the idc translator is called.
    //----------------------------------------
    IutpToIcdPackageTranslator m_icdPackageForIdcTranslator;

    //========================================
    //! \brief Data to idc package translator.
    //!
    //! All successfully translated packages are published to the device.
    //----------------------------------------
    IcdToIdcPackageTranslator m_idcPackageForDeviceTranslator;

    //========================================
    //! \brief Current UDP ECU configuration.
    //----------------------------------------
    MvisEcuUdpConfigurationPtr m_configuration;

    //========================================
    //! \brief Locked UDP ECU configuration.
    //----------------------------------------
    MvisEcuUdpConfigurationPtr m_lockedConfiguration;
}; // class MvisEcuUdpDevice

//==============================================================================
//! \brief Pointer type of MVIS ECU UDP device.
//------------------------------------------------------------------------------
using MvisEcuUdpDevicePtr = std::unique_ptr<MvisEcuUdpDevice>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
