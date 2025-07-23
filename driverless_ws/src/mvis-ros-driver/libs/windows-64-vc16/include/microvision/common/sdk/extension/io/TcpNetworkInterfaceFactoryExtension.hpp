//==============================================================================
//! \file
//!
//! \brief Extension which provides TCP implementations for the network interface factory.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Feb 24, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/extension/NetworkInterfaceFactoryExtension.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Extension which provides TCP implementations for the network interface factory.
//! \extends NetworkInterfaceFactoryExtension
//------------------------------------------------------------------------------
class TcpNetworkInterfaceFactoryExtension : public NetworkInterfaceFactoryExtension
{
public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    TcpNetworkInterfaceFactoryExtension();

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~TcpNetworkInterfaceFactoryExtension() override;

public:
    //========================================
    //! \brief Prefer this extensions in call order.
    //! \returns Either \c true if this extension should be preferred otherwise \c false.
    //----------------------------------------
    bool isToPrefer() const override;

    //========================================
    //! \brief Get list of supported network interface types.
    //! \param[in] configurationType  (optional) If not empty, the network interface
    //!                               types in the returned list are restricted to those
    //!                               which support this configuration type.
    //! \return (Filtered) List with all supported network interface types.

    //----------------------------------------
    std::vector<std::string> getNetworkInterfaceTypes(const std::string& configurationType
                                                      = std::string{}) const override;

    //========================================
    //! \brief Create a NetworkInterface from type.
    //! \param[in] networkInterfaceType  Type of the network interface to be created.
    //! \param[in] configuration         (optional) Network configuration.
    //! \return Either a NetworkInterface pointer if registered, otherwise \c nullptr.
    //! \note If the network configuration is not compatible with implementation it returns \c nullptr.
    //----------------------------------------
    NetworkInterfaceUPtr createNetworkInterface(const std::string& networkInterfaceType,
                                                const NetworkConfigurationPtr& configuration = nullptr) const override;

}; // class TcpNetworkInterfaceFactoryExtension

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
