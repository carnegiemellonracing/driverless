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
//! \date Mar 10, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/devices/MvisEcuConfiguration.hpp>
#include <microvision/common/sdk/devices/Commander.hpp>
#include <microvision/common/sdk/devices/IdcDevice.hpp>
#include <microvision/common/sdk/devices/MvisTrackingBox.hpp>
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

//=================================================
//! \brief The wrapper for the MVIS TrackingBox device to adapt it to the new SDK 7 device interface.
//!
//! \extends IdcDevice
//! \extends Commander
//-------------------------------------------------
class MvisTrackingBoxDevice : public IdcDevice, public Commander, private IdcDataPackageListener
{
public:
    //========================================
    //! \brief Type name of this device.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

    //========================================
    //! \brief Default tcp port.
    //----------------------------------------
    static constexpr MICROVISION_SDK_API uint16_t defaultTcpPort{12002};

private:
    //========================================
    //! \brief  Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::MvisTrackingBoxDevice";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Constructor
    //----------------------------------------
    MvisTrackingBoxDevice();

    //========================================
    //! \brief Destructor
    //----------------------------------------
    ~MvisTrackingBoxDevice() override;

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
    //! \sa MvisEcuConfiguration
    //----------------------------------------
    ConfigurationPtr getDeviceConfiguration() const override;

    //========================================
    //! \brief Set the configuration of this device.
    //! \param[in] deviceConfiguration  The shared pointer to new configuration of this device.
    //! \return Either \c true if this device can be configured with those configuration, otherwise \c false.
    //!
    //! \sa MvisEcuConfiguration
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
    //! \param[in] callback     (Optional) Will called with result of command sending. Per default nullptr.
    //----------------------------------------
    void sendCommand(const CommandPtr& command,
                     const ExporterPtr& exporter,
                     const CallbackType& callback = nullptr) override;

    //========================================
    //! \brief Send a command which expects no reply.
    //! \param[in]      command         Command to be sent.
    //! \param[in]      exporter        Exporter to serialize command.
    //! \param[in, out] reply           The reply container for the reply to be stored into.
    //! \param[in]      timeoutInMs     (Optional) Number of milliseconds to wait for a reply. Per default 500.
    //! \param[in]      callback        (Optional) Will called with result of command sending. Per default nullptr.
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
    //! \brief Callback function to adapt \m_TrackingBox to the 'data type range vector' property of the configuration.
    //!
    //! \param[in] property    The 'data type range vector' property.
    //----------------------------------------
    void onConfigurationPropertyRangeVectorChanged(const ConfigurationProperty& property, const Any&);

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
    MvisEcuConfigurationPtr m_configuration;

    //========================================
    //! \brief Locked configuration.
    //----------------------------------------
    MvisEcuConfigurationPtr m_lockedConfiguration;

    //========================================
    //! \brief Obsolete MvisTrackingBox device which is wrapped in this class.
    //----------------------------------------
    std::shared_ptr<MvisTrackingBox> m_delegate;

    //========================================
    //! \brief Async command results.
    //----------------------------------------
    std::list<std::future<void>> m_commandResults;
}; // class MvisTrackingBoxDevice

//==============================================================================
//! \brief Pointer type of TrackingBox device.
//------------------------------------------------------------------------------
using MvisTrackingBoxDevicePtr = std::unique_ptr<MvisTrackingBoxDevice>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

// Reset compiler warning settings to default.
ALLOW_WARNINGS_END
