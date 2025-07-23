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

#include <microvision/common/sdk/config/io/TcpConfiguration.hpp>
#include <microvision/common/sdk/devices/Commander.hpp>
#include <microvision/common/sdk/devices/IdcDevice.hpp>
#include <microvision/common/sdk/devices/MvisMiniLux.hpp>
#include <microvision/common/sdk/io/idc/DataPackageToIdcTranslator.hpp>
#include <microvision/common/sdk/listener/IdcDataPackageListener.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/logging/logging.hpp>

#include <future>
#include <list>

//==============================================================================

// Change the compiler warning settings until ALLOW_WARNINGS_END.
ALLOW_WARNINGS_BEGIN
// Allow deprecated warnings in deprecated code to avoid
// compiler errors because of deprecated dependencies.
ALLOW_WARNING_DEPRECATED

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Class to connect to a MiniLux Sensor.
//!
//! The configuration of the LUX sensor device is basically the default tcp configuration.
//! So every device receives only data from one network source.
//!
//! \extends IdcDevice
//! \extends Commander
//------------------------------------------------------------------------------
class MvisMiniLuxSensorDevice final : public IdcDevice, public Commander, private IdcDataPackageListener
{
public:
    //========================================
    //! \brief Type name of this device
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

    //========================================
    //! \brief Default tcp port.
    //----------------------------------------
    static constexpr MICROVISION_SDK_API uint16_t defaultTcpPort{12006};

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::MvisMiniLuxSensorDevice";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Constructor.
    //----------------------------------------
    MvisMiniLuxSensorDevice();

    //========================================
    //! \brief Destructor.
    //----------------------------------------
    ~MvisMiniLuxSensorDevice() override;

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
    //! \sa TcpConfiguration
    //----------------------------------------
    ConfigurationPtr getDeviceConfiguration() const override;

    //========================================
    //! \brief Set the configuration of this device.
    //! \param[in] deviceConfiguration  The shared pointer to new configuration of this device.
    //! \return Either \c true if this device can be configured with those configuration, otherwise \c false.
    //!
    //! \sa TcpConfiguration
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

public: // implements Commander
    //========================================
    //! \brief Send a command which expects no reply.
    //! \param[in] command      Command to be sent.
    //! \param[in] exporter     Exporter to serialize command.
    //! \param[in] callback     (Optional) Will called with result of command sending.
    //!                         Per default nullptr.
    //----------------------------------------
    void sendCommand(const CommandPtr& command,
                     const ExporterPtr& exporter,
                     const CallbackType& callback = nullptr) override;

    //========================================
    //! \brief Send a command which expects no reply.
    //! \param[in]      command         Command to be sent.
    //! \param[in]      exporter        Exporter to serialize command.
    //! \param[in, out] reply           The reply container for the reply to be stored into.
    //! \param[in]      timeoutInMs     (Optional) Number of milliseconds to wait for a reply.
    //! \param[in]      callback        (Optional) Will called with result of command sending.
    //!                                 Per default nullptr.
    //----------------------------------------
    void sendCommand(const CommandPtr& command,
                     const ExporterPtr& exporter,
                     ReplyPtr& reply,
                     const uint32_t timeoutInMs   = 500U,
                     const CallbackType& callback = nullptr) override;

private: // implements IdcDataPackageListener
    //========================================
    //! \brief Method to be called if a new IdcDataPackage has been received.
    //! \param[in] data  Shared idc data package pointer of received data.
    //----------------------------------------
    void onDataReceived(const IdcDataPackagePtr& data) override;

private:
    //========================================
    //! \brief Clean up the command result queue.
    //! \param[in] keepIfNotComplete  (Optional) Keep in queue if command is not complete, default \c true.
    //----------------------------------------
    void cleanupCommandResults(const bool keepIfNotComplete = true);

private:
    //========================================
    //! \brief The Uri which denotes the packets dedicated for this device.
    //----------------------------------------
    Uri m_packageUri;

    //========================================
    //! \brief  Increment index of received packages.
    //----------------------------------------
    ThreadSafe<int64_t> m_packageIndex;

    //========================================
    //! \brief Reassembles data packages sent to this device to regular idc data packages.
    //----------------------------------------
    DataPackageToIdcTranslator m_idcPackageForDeviceTranslator;

    //========================================
    //! \brief Current configuration.
    //----------------------------------------
    TcpConfigurationPtr m_configuration;

    //========================================
    //! \brief Locked configuration.
    //----------------------------------------
    TcpConfigurationPtr m_lockedConfiguration;

    //========================================
    //! \brief Obsolete MvisMiniLux device which is wrapped in this class.
    //----------------------------------------
    std::unique_ptr<MvisMiniLux> m_delegate;

    //========================================
    //! \brief Async command results.
    //----------------------------------------
    std::list<std::future<void>> m_commandResults;
}; // class MvisMiniLuxSensorDevice

//==============================================================================
//! \brief Pointer type of LUX sensor device.
//------------------------------------------------------------------------------
using MvisMiniLuxSensorDevicePtr = std::unique_ptr<MvisMiniLuxSensorDevice>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

// Reset compiler warning settings to default.
ALLOW_WARNINGS_END
