//==============================================================================
//! \file
//!
//! \brief Interface for UDP io network resources.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Sep 30, 2019
//------------------------------------------------------------------------------

#pragma once

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/misc/ThreadSafe.hpp>

#include <microvision/common/sdk/config/io/UdpConfiguration.hpp>
#include <microvision/common/sdk/io/NetworkInterface.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Interface for UDP io network resources.
//!
//! Implement this interface to provide UDP network resource handling.
//!
//! \extends NetworkInterface
//------------------------------------------------------------------------------
class UdpBase : public NetworkInterface
{
public:
    //========================================
    //! \brief Maximum size of a UDP datagram.
    //!
    //! 65.535 UDP - 8 Byte UDP Header - 20 Byte IPv4 Header
    //----------------------------------------
    static constexpr uint16_t maxUdpDatagramSize{65507U};

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    UdpBase();

    //========================================
    //! \brief Move constructor disabled.
    //----------------------------------------
    UdpBase(UdpBase&& toMove) = delete;

    //========================================
    //! \brief Copy constructor disabled.
    //----------------------------------------
    UdpBase(const UdpBase& toCopy) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~UdpBase() override = default;

public:
    //========================================
    //! \brief Move assignment operator disabled.
    //----------------------------------------
    UdpBase& operator=(UdpBase&& toMove) = delete;

    //========================================
    //! \brief Copy assignment operator disabled.
    //----------------------------------------
    UdpBase& operator=(const UdpBase& toCopy) = delete;

public: // implements Configurable
    //========================================
    //! \brief Get supported types of configuration.
    //!
    //! Configuration type is a human readable unique string name of the configuration
    //! used to address it in code.
    //!
    //! \returns All supported configuration types.
    //----------------------------------------
    const std::vector<std::string>& getConfigurationTypes() const override;

public: // implements NetworkBase
    //========================================
    //! \brief Get the source/destination Uri.
    //! \return Describing source/destination Uri of stream.
    //----------------------------------------
    Uri getUri() const override;

public: // implements NetworkInterface
    //========================================
    //! \brief Get pointer to network configuration which is used to establishing the connection.
    //!
    //! The pointer points to an implementation of NetworkConfiguration,
    //! for example TcpConfiguration or UdpConfiguration.
    //!
    //! \return Pointer to an instance of NetworkConfiguration.
    //----------------------------------------
    NetworkConfigurationPtr getConfiguration() const override;

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
    bool setConfiguration(const NetworkConfigurationPtr& configuration) override;

    //========================================
    //! \brief Get handle to sync thread resources of network interfaces.
    //! \return Either pointer to sync handle or \c nullptr if not synced.
    //----------------------------------------
    ThreadSyncPtr getSynchronizationHandle() const override;

    //========================================
    //! \brief Set handle to sync thread resources of network interfaces.
    //! \param[in] syncHandle  Pointer to sync handle to enable sync or \c nullptr to disable sync.
    //----------------------------------------
    void setSynchronizationHandle(const ThreadSyncPtr& syncHandle) override;

public:
    //========================================
    //! \brief Get configuration of UDP network resource.
    //! \return Either a \c UdpConfiguration shared pointer if set or otherwise nullptr.
    //----------------------------------------
    UdpConfigurationPtr getUdpConfiguration() const;

    //========================================
    //! \brief Set configuration of network resource.
    //! \param[in] configuration  Shared pointer to new configuration of network resource.
    //! \return Either \c true if UDP receiver/sender can be configured with those configuration, otherwise \c false.
    //----------------------------------------
    bool setUdpConfiguration(const UdpConfigurationPtr& configuration);

protected:
    //========================================
    //! \brief Make Uri from UdpConfiguration.
    //! \param[in] config  UdpConfiguration to make network uri.
    //! \return Describing source/destination Uri.
    //----------------------------------------
    Uri makeUri(const UdpConfigurationPtr& config) const;

private:
    //========================================
    //! \brief Udp configuration.
    //----------------------------------------
    UdpConfigurationPtr m_configuration;

    //========================================
    //! \brief Thread resources sync handler.
    //----------------------------------------
    ThreadSyncPtr m_syncHandle;
}; // class UdpBase

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
