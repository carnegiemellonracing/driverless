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
//! \date Feb 6, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/Configuration.hpp>
#include <microvision/common/sdk/config/Configurable.hpp>
#include <microvision/common/sdk/io/NetworkBase.hpp>

#include <microvision/common/sdk/listener/Emitter.hpp>
#include <boost/asio.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface which provides general device control.
//! \extends Configurable
//! \extends Emitter
//------------------------------------------------------------------------------
class Device : public Configurable, public virtual Emitter
{
public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~Device() override = default;

public:
    //========================================
    //! \brief Get the configuration type of the device.
    //! \returns Configuration type of this device.
    //----------------------------------------
    virtual const std::string& getDeviceType() const = 0;

public:
    //========================================
    //! \brief Get the last error which is caught in network io thread.
    //! \returns Exception pointer if error caught, otherwise nullptr.
    //----------------------------------------
    virtual std::exception_ptr getLastError() const = 0;

    //========================================
    //! \brief Checks that a connection is established.
    //! \returns Either \c true if connection is established or otherwise \c false.
    //----------------------------------------
    virtual bool isConnected() const = 0;

    //========================================
    //! \brief Checks that a connection is established and packages are still processed.
    //! \returns Either \c true if connection is established or work is ongoing on received data or otherwise \c false.
    //----------------------------------------
    virtual bool isWorking() = 0;

public:
    //========================================
    //! \brief Establish a connection to the network resource.
    //----------------------------------------
    virtual void connect() = 0;

    //========================================
    //! \brief Disconnect from the network resource.
    //----------------------------------------
    virtual void disconnect() = 0;

public:
    //========================================
    //! \brief Get configuration of device.
    //! \return Either a \c Configuration shared pointer if set or otherwise nullptr.
    //----------------------------------------
    virtual ConfigurationPtr getDeviceConfiguration() const = 0;

    //========================================
    //! \brief Set configuration of device.
    //! \param[in] deviceConfiguration  Shared pointer to new configuration of device.
    //! \return Either \c true if device can be configured with those configuration, otherwise \c false.
    //----------------------------------------
    virtual bool setDeviceConfiguration(const ConfigurationPtr& deviceConfiguration) = 0;

public:
    //========================================
    //! \brief Get whether a configuration is mandatory for this Configurable.
    //! \return \c true if a configuration is mandatory for this Configurable,
    //!         \c false otherwise.
    //----------------------------------------
    bool isConfigurationMandatory() const final { return true; }

public:
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
    virtual bool lockConfiguration() = 0;

    //========================================
    //! \brief Released locked configuration.
    //!
    //! This does not work while the device is connected.
    //!
    //! \return Either \c true if the configuration is unlocked, otherwise \c false.
    //----------------------------------------
    virtual bool unlockConfiguration() = 0;

public:
    //========================================
    //! \brief Process a data package as sensor input from another source (e.g. as read from a file).
    //! \param[in] dataPackage  The data package to process.
    //! \return Either \c true if the processing was successful, otherwise \c false.
    //----------------------------------------
    virtual bool processDataPackage(const DataPackagePtr& dataPackage) = 0;

}; // class Device

//==============================================================================
//! \brief Nullable Device pointer.
//------------------------------------------------------------------------------
using DevicePtr = std::unique_ptr<Device>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
