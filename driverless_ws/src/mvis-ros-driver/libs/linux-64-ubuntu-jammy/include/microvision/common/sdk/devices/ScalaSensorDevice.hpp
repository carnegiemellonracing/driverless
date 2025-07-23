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
//! \date Mar 03, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/io/NetworkConfiguration.hpp>
#include <microvision/common/sdk/devices/IdcDevice.hpp>
#include <microvision/common/sdk/devices/Scala.hpp>
#include <microvision/common/sdk/io/idc/DataPackageToIdcTranslator.hpp>
#include <microvision/common/sdk/listener/IdcDataPackageListener.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/logging/logging.hpp>

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
//! \brief Access Scala sensor device.
//!
//! The configuration of the Scala sensor device is basically the default UDP/TCP configuration.
//! So every device receives only data from one network source.
//!
//! \extends IdcDevice
//------------------------------------------------------------------------------
class ScalaSensorDevice final : public IdcDevice, private IdcDataPackageListener
{
public:
    //========================================
    //! \brief Type name of this device.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

    //========================================
    //! \brief Default tcp port.
    //----------------------------------------
    static constexpr MICROVISION_SDK_API uint16_t defaultTcpPort{12004};

private:
    //========================================
    //! \brief Logger name for configuration setup.
    //----------------------------------------
    static constexpr const char* m_loggerId = "microvision::common::sdk::ScalaSensorDevice";

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    ScalaSensorDevice();

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~ScalaSensorDevice() override;

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

private: // implements IdcDataPackageListener
    //========================================
    //! \brief Method to be called if a new IdcDataPackage has been received.
    //! \param[in] data  Shared idc data package pointer of received data.
    //----------------------------------------
    void onDataReceived(const IdcDataPackagePtr& data) override;

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
    NetworkConfigurationPtr m_configuration;

    //========================================
    //! \brief Locked configuration.
    //----------------------------------------
    NetworkConfigurationPtr m_lockedConfiguration;

    //========================================
    //! \brief Obsolete Scala device which is wrapped in this class.
    //----------------------------------------
    std::unique_ptr<Scala> m_delegate;

}; // class ScalaSensorDevice

//==============================================================================
//! \brief Pointer type of scala sensor device.
//------------------------------------------------------------------------------
using ScalaSensorDevicePtr = std::unique_ptr<ScalaSensorDevice>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

// Reset compiler warning settings to default.
ALLOW_WARNINGS_END
