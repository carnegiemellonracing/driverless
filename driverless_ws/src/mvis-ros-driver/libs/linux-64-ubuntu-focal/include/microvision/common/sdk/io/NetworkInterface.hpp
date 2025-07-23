//==============================================================================
//! \file
//!
//! \brief Extended interface for general io network resources.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jul 07, 2021
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/io/NetworkConfiguration.hpp>
#include <microvision/common/sdk/config/Configurable.hpp>
#include <microvision/common/sdk/misc/ThreadSync.hpp>
#include <microvision/common/sdk/io/NetworkBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Extended interface for general io network resources.
//!
//! Implement this interface to provide general network resource handling.
//! This is necessary if you wanna implement a NetworkInterfaceExtension.
//------------------------------------------------------------------------------
class NetworkInterface : public virtual NetworkBase, public Configurable
{
public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~NetworkInterface() override = default;

public:
    //========================================
    //! \brief Get network interface type which is used to identify the implementation.
    //!
    //! Network interface type is a human readable unique string name of the
    //! NetworkInterface used to address it in code.
    //!
    //! \returns Human readable unique name of network interface implementation.
    //----------------------------------------
    virtual const std::string& getType() const = 0;

    //========================================
    //! \brief Get all errors which are caught in network io threads.
    //! \returns List of exception pointers.
    //----------------------------------------
    virtual std::vector<std::exception_ptr> getErrors() const = 0;

    //========================================
    //! \brief Get pointer to network configuration which is used to establishing the connection.
    //!
    //! The pointer points to an implementation of NetworkConfiguration,
    //! for example TcpConfiguration or UdpConfiguration.
    //!
    //! \return Pointer to an instance of NetworkConfiguration.
    //----------------------------------------
    virtual NetworkConfigurationPtr getConfiguration() const = 0;

    //========================================
    //! \brief Set pointer to network configuration which is used to establishing the connection.
    //!
    //! The pointer points to an implementation of NetworkConfiguration,
    //! for example TcpConfiguration or UdpConfiguration.
    //!
    //! \param[in] configuration  Pointer to an instance of NetworkConfiguration.
    //! \return Either \c true if the configuration is supported by implementation or otherwise \c false.
    //! \note If the configuration is not supported by implementation it will not change the current value.
    //!       However, if \a configuration is \c nullptr the configuration of NetworkInterface will be reset.
    //----------------------------------------
    virtual bool setConfiguration(const NetworkConfigurationPtr& configuration) = 0;

    //========================================
    //! \brief Get handle to sync thread resources of network interfaces.
    //! \return Either pointer to sync handle or \c nullptr if not synced.
    //----------------------------------------
    virtual ThreadSyncPtr getSynchronizationHandle() const = 0;

    //========================================
    //! \brief Set handle to sync thread resources of network interfaces.
    //! \param[in] syncHandle  Pointer to sync handle to enable sync or \c nullptr to disable sync.
    //----------------------------------------
    virtual void setSynchronizationHandle(const ThreadSyncPtr& syncHandle) = 0;

public:
    //========================================
    //! \brief Get whether a configuration is mandatory for this Configurable.
    //! \return \c true if a configuration is mandatory for this Configurable,
    //!         \c false otherwise.
    //----------------------------------------
    bool isConfigurationMandatory() const final { return true; }
}; // class NetworkInterface

//==============================================================================
//! \brief Nullable NetworkInterface pointer.
//------------------------------------------------------------------------------
using NetworkInterfaceUPtr = std::unique_ptr<NetworkInterface>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
