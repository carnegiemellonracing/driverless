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
//! \date Jan 30, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/io/NetworkConfiguration.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Udp network configuration.
//!
//! This configuration contains all the required properties for an udp connection.
//!
//! Example udp configuration:
//! \code
//! auto config = ConfigurationFactory::getInstance().createConfiguration(UdpConfiguration::typeName);
//! config->trySetValue("multicast_ip", makeIp("239.1.2.5")); // if false: configuration property does not exists or type is incompatible!
//! config->trySetValue("broadcast_ip", makeIp("192.168.1.255")); // if false: configuration property does not exists or type is incompatible!
//! config->trySetValue("local_port", uint16_t{12349}); // if false: configuration property does not exists or type is incompatible!
//! \endcode
//!
//! New configuration properties added:
//! Property Name  | Type         | Description                        | Default
//! -------------- | ------------ | ---------------------------------- | -------------
//! multicast_ip   | IpAddressPtr | Ip multicast address               | nullptr
//! broadcast_ip   | IpAddressPtr | Ip broadcast address for UdpSender | nullptr
//!
//! \sa NetworkConfiguration
//------------------------------------------------------------------------------
class UdpConfiguration : public NetworkConfiguration
{
public:
    //========================================
    //! \brief Configuration type name.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

    //==============================================================================
    //! \brief Unique config id for property of 'udp multicast address'.
    //------------------------------------------------------------------------------
    static const MICROVISION_SDK_API std::string multicastIpAddressConfigId;

    //==============================================================================
    //! \brief Unique config id for property of 'udp broadcast address'.
    //------------------------------------------------------------------------------
    static const MICROVISION_SDK_API std::string broadcastIpAddressConfigId;

public:
    //========================================
    //! \brief Get name of type of this configuration.
    //! \returns Configuration type.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string& getTypeName();

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    UdpConfiguration();

    //========================================
    //! \brief Construct and update udp network properties with optional default values.
    //! \param[in] defaultLocalIpAddress      Default value for local IP address.
    //! \param[in] defaultLocalPort           (Optional) Default value for local port, default is \c 0.
    //! \param[in] defaultRemoteIpAddress     (Optional) Default value for remote IP address, default is \c nullptr.
    //! \param[in] defaultRemotePort          (Optional) Default value for remote port, default is \c 0.
    //! \param[in] defaultTimeoutInMs         (Optional) Default value for timeout in milliseconds, default is \c 500U.
    //! \param[in] defaultBufferSize          (Optional) Default value for buffer size, default is \c 0.
    //! \param[in] defaultMulticastIpAddress  (Optional) Default value for multicast IP address, default is \c nullptr.
    //! \param[in] defaultBroadcastIpAddress  (Optional) Default value for broadcast IP address for UdpSender, default is \c nullptr.
    //----------------------------------------
    UdpConfiguration(const IpAddressPtr defaultLocalIpAddress,
                     const uint16_t defaultLocalPort              = 0U,
                     const IpAddressPtr defaultRemoteIpAddress    = nullptr,
                     const uint16_t defaultRemotePort             = 0U,
                     const uint32_t defaultTimeoutInMs            = defaultTimeoutInMsConstant,
                     const uint32_t defaultBufferSize             = 0U,
                     const IpAddressPtr defaultMulticastIpAddress = nullptr,
                     const IpAddressPtr defaultBroadcastIpAddress = nullptr);

    //========================================
    //! \brief Copy constructor to copy and update udp network properties.
    //! \param[in] other  Other UdpConfiguration to copy.
    //----------------------------------------
    UdpConfiguration(const UdpConfiguration& other);

    //========================================
    //! \brief Disabled move constructor because of thread-safe guarantee.
    //----------------------------------------
    UdpConfiguration(UdpConfiguration&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~UdpConfiguration() override;

public: // implements Configuration
    //========================================
    //! \brief Get type of configuration to match with.
    //! \returns Configuration type.
    //----------------------------------------
    const std::string& getType() const override;

    //========================================
    //! \brief Get copy of configuration.
    //! \returns Pointer to new copied Configuration.
    //----------------------------------------
    ConfigurationPtr copy() const override;

public:
    //========================================
    //! \brief Get IP multicast address configuration property.
    //! \returns Ip multicast address configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<IpAddressPtr>& getMulticastIpAddress();

    //========================================
    //! \brief Get IP broadcast address configuration property.
    //! \returns Ip broadcast address configuration property.
    //!
    //! \note Used by UdpSender only.
    //----------------------------------------
    ConfigurationPropertyOfType<IpAddressPtr>& getBroadcastIpAddress();

protected: // implements NetworkConfiguration
    //========================================
    //! \brief Get UriProtocol of network configuration.
    //! \returns UriProtocol of network configuration.
    //----------------------------------------
    UriProtocol getUriProtocol() const override;

    //========================================
    //! \brief Validate source uri would be acceptable by configuration.
    //! \param[in] source  Source Uri to be validated.
    //! \returns Either \c true if source Uri is acceptable, otherwise \c false.
    //----------------------------------------
    bool acceptPackageSource(const Uri& source) const override;

    //========================================
    //! \brief Validate local and remote uri would be acceptable by configuration.
    //! \param[in] local   Local Uri to be validate.
    //! \param[in] remote  Remote Uri to be validate.
    //! \returns Either \c true if local and remote Uri is acceptable, otherwise \c false.
    //----------------------------------------
    bool acceptPackageSource(const Uri& local, const Uri& remote) const override;

private:
    //========================================
    //! \brief Ip multicast address configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<IpAddressPtr> m_multicastIpAddress;

    //========================================
    //! \brief Ip broadcast address configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<IpAddressPtr> m_broadcastIpAddress;

}; // class UdpConfiguration

//==============================================================================
//! \brief Nullable UdpConfiguration pointer.
//------------------------------------------------------------------------------
using UdpConfigurationPtr = std::shared_ptr<UdpConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
