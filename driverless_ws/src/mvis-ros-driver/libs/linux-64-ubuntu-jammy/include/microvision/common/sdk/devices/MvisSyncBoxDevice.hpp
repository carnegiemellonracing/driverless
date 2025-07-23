//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 05, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/devices/MvisSyncBoxConfiguration.hpp>
#include <microvision/common/sdk/devices/IdcDevice.hpp>

#include <microvision/common/sdk/io/idc/DataPackageToIdcTranslator.hpp>
#include <microvision/common/sdk/extension/NetworkInterfaceFactory.hpp>
#include <microvision/common/sdk/misc/BackgroundWorker.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/logging/logging.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief A device for handling data from a MVIS SyncBox.
//!
//! \extends IdcDevice
//------------------------------------------------------------------------------
class MvisSyncBoxDevice final : public IdcDevice
{
public:
    //========================================
    //! \brief Type name of this device
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::MvisSyncBoxDevice";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    MvisSyncBoxDevice();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~MvisSyncBoxDevice() override;

public: // implements Configurable
    //========================================
    //! \brief Get supported types of configuration.
    //!
    //! Supported means that this device can interpret imported data packages using this configuration.
    //! Configuration type is a human readable unique string name of the configuration
    //! used to address it in code.
    //!
    //! \return All supported configuration types.
    //----------------------------------------
    const std::vector<std::string>& getConfigurationTypes() const override;

public: // implements Device
        //========================================
        //! \brief Get the type name of this device.
        //! \return The type name of this device.
        //----------------------------------------
    const std::string& getDeviceType() const override;

    //========================================
    //! \brief Get the last error which has occurred.
    //! \return The exception of the last occurred error.
    //----------------------------------------
    std::exception_ptr getLastError() const override;

    //========================================
    //! \brief Checks if a connection has been established.
    //! \return Either \c true if connection is established or otherwise \c false.
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
    //! \return The current configuration.
    //!
    //! \sa MvisSyncBoxConfiguration
    //----------------------------------------
    ConfigurationPtr getDeviceConfiguration() const override;

    //========================================
    //! \brief Set the configuration of this device.
    //! \param[in] deviceConfiguration  The shared pointer to new configuration of this device.
    //! \return Either \c true if this device can be configured with those configuration, otherwise \c false.
    //!
    //! \sa MvisSyncBoxConfiguration
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
    //! \brief Method to be called if a new IdcDataPackage has been received.
    //! \param[in] data  Shared idc data package pointer of received data.
    //----------------------------------------
    void notifyListeners(const IdcDataPackagePtr& data);

    //========================================
    //! \brief Main method of thread which observes the receiver.
    //! \param[in] worker  Executing Background worker m_observerWorker.
    //! \return Either \c true too keep thread alive or otherwise \c false.
    //----------------------------------------
    bool observerMain(BackgroundWorker& worker);

private:
    //========================================
    //! \brief Increment index of received packages.
    //----------------------------------------
    ThreadSafe<int64_t> m_packageIndex;

    //========================================
    //! \brief Receiver observer thread.
    //----------------------------------------
    BackgroundWorker m_observerWorker;

    //========================================
    //! \brief Receiver for udp data packages.
    //----------------------------------------
    NetworkInterfaceUPtr m_udpReceiver;

    //========================================
    //! \brief Reassembles data packages sent to this device to regular idc data packages.
    //----------------------------------------
    DataPackageToIdcTranslator m_idcPackageForDeviceTranslator;

    //========================================
    //! \brief Current configuration.
    //----------------------------------------
    SyncBoxConfigurationPtr m_configuration;

    //========================================
    //! \brief Locked configuration.
    //----------------------------------------
    SyncBoxConfigurationPtr m_lockedConfiguration;

}; // class MvisSyncBoxDevice

//==============================================================================
//! \brief Pointer type of SyncBox sensor device.
//------------------------------------------------------------------------------
using MvisSyncBoxSensorDevicePtr = std::unique_ptr<MvisSyncBoxDevice>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
