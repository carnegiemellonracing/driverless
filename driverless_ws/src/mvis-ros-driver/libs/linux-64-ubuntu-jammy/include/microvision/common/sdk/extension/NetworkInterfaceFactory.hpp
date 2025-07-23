//==============================================================================
//! \file
//!
//! \brief Factory for NetworkInterface.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jul 06, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/extension/NetworkInterfaceFactoryExtension.hpp>
#include <microvision/common/sdk/extension/Extendable.hpp>

#include <microvision/common/logging/logging.hpp>

#include <memory>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Factory for NetworkInterface.
//!
//! This singleton factory is extendable and will create a NetworkInterface.
//! The NetworkInterface provide implementation of different protocols of communication.
//!
//! \extends Extendable<NetworkInterfaceFactoryExtension>
//------------------------------------------------------------------------------
class NetworkInterfaceFactory final : public Extendable<NetworkInterfaceFactoryExtension>
{
private:
    //========================================
    //! \brief Logger name for setup logger configuration.
    //----------------------------------------
    static constexpr const char* m_loggerId{"microvision::common::sdk::NetworkInterfaceFactory"};

    //========================================
    //! \brief Provides common logger interface.
    //! \sa microvision::common::logging::Logger
    //----------------------------------------
    static microvision::common::logging::LoggerSPtr m_logger;

private:
    //========================================
    //! Constructor registering all MVIS SDK network interface extensions.
    //!
    //! Network interface extensions which are not delivered as plugins
    //! to the customer have to be registered manually.
    //!
    //! \note When adding new network interface extensions with the sdk
    //!       they have to be registered here.
    //----------------------------------------
    NetworkInterfaceFactory();

public:
    //========================================
    //! \brief Get the singleton instance of NetworkInterfaceFactory.
    //! \return Singleton instance of NetworkInterfaceFactory.
    //----------------------------------------
    static NetworkInterfaceFactory& getInstance();

public:
    //========================================
    //! \brief Get list of supported network interface types.
    //! \param[in] configurationType  (optional) If not empty, the network interface
    //!                               types in the returned list are restricted to those
    //!                               which support this configuration type.
    //! \return (Filtered) List with all supported network interface types.

    //----------------------------------------
    std::vector<std::string> getNetworkInterfaceTypes(const std::string& configurationType = std::string{}) const;

    //========================================
    //! \brief Create a NetworkInterface from type.
    //! \param[in] networkInterfaceType  Type of the network interface to be created.
    //! \param[in] configuration         (optional) Network configuration.
    //! \return Either a NetworkInterface pointer if registered, otherwise \c nullptr.
    //! \note If the network configuration is not compatible with implementation it returns \c nullptr.
    //----------------------------------------
    NetworkInterfaceUPtr createNetworkInterface(const std::string& networkInterfaceType,
                                                const NetworkConfigurationPtr& configuration = nullptr) const;

}; // class NetworkInterfaceFactory

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
