//==============================================================================
//! \file
//!
//! \brief Interface to extend network interface factory.
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

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/io/NetworkInterface.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface to extend network interface factory.
//!
//! This interface will provide functionality to create network interfaces.
//! A derived extension is then registered at the factory to enable creation
//! of network interfaces of the type the extension provides.
//!
//! \sa NetworkInterfaceFactory
//------------------------------------------------------------------------------
class NetworkInterfaceFactoryExtension
{
public:
    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    virtual ~NetworkInterfaceFactoryExtension() = default;

public:
    //========================================
    //! \brief Prefer this extensions in call order.
    //! \note This is only used to override existing extensions for test mockups.
    //! \return Either \c true if this extension should be preferred otherwise \c false.
    //----------------------------------------
    virtual bool isToPrefer() const = 0;

    //========================================
    //! \brief Get list of supported network interface types.
    //! \param[in] configurationType  (optional) If not empty, the network interface
    //!                               types in the returned list are restricted to those
    //!                               which support this configuration type.
    //! \return (Filtered) List with all supported network interface types.

    //----------------------------------------
    virtual std::vector<std::string> getNetworkInterfaceTypes(const std::string& configurationType
                                                              = std::string{}) const = 0;

    //========================================
    //! \brief Create a NetworkInterface from type.
    //! \param[in] networkInterfaceType  Type of the network interface to be created.
    //! \param[in] configuration         (optional) Network configuration.
    //! \return Either \c NetworkInterface pointer if registered, otherwise \c nullptr.
    //! \note If the network configuration is not compatible with implementation it returns \c nullptr.
    //----------------------------------------
    virtual NetworkInterfaceUPtr createNetworkInterface(const std::string& networkInterfaceType,
                                                        const NetworkConfigurationPtr& configuration
                                                        = nullptr) const = 0;

}; // class NetworkInterfaceFactoryExtension

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
