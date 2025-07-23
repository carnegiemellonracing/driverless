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

#include <microvision/common/sdk/config/ConfigurationPropertyOfType.hpp>
#include <microvision/common/sdk/config/Configuration.hpp>
#include <microvision/common/sdk/io/NetworkDataPackage.hpp>
#include <microvision/common/sdk/io/ip/IpAddress.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Abstract base for network connection configuration.
//!
//! Example network configuration:
//! \code
//! auto config = ConfigurationFactory::getInstance().createConfiguration(NetworkConfiguration::typeName);
//! config->trySetValue("remote_ip", makeIp("239.1.2.5")); // if false: configuration property does not exists or type is incompatible!
//! config->trySetValue("remote_port", uint16_t{12349}); // if false: configuration property does not exists or type is incompatible!
//! \endcode
//!
//! Configuration properties:
//!
//! Property Name  | Type         | Description            | Default
//! -------------- | ------------ | ---------------------- | -------------
//! local_ip       | IpAddressPtr | Local IP address       | nullptr
//! local_port     | uint16_t     | Local port             | 0
//! remote_ip      | IpAddressPtr | Remote IP address      | nullptr
//! remote_port    | uint16_t     | Remote port            | 0
//! timeout_in_ms  | uint32_t     | Timeout ms             | 500U
//! buffer_size    | uint32_t     | Buffer size            | 0
//!
//! \extends microvision::common::sdk::Configuration
//------------------------------------------------------------------------------
class NetworkConfiguration : public virtual Configuration
{
public:
    //==============================================================================
    //! \brief Unique config id for property of 'local IP address'.
    //------------------------------------------------------------------------------
    static const MICROVISION_SDK_API std::string localIpAddressConfigId;

    //==============================================================================
    //! \brief Unique config id for property of 'local port'.
    //------------------------------------------------------------------------------
    static MICROVISION_SDK_API const std::string localPortConfigId;

    //==============================================================================
    //! \brief Unique config id for property of 'remote IP address'.
    //------------------------------------------------------------------------------
    static MICROVISION_SDK_API const std::string remoteIpAddressConfigId;

    //==============================================================================
    //! \brief Unique config id for property of 'remote port'.
    //------------------------------------------------------------------------------
    static MICROVISION_SDK_API const std::string remotePortConfigId;

    //==============================================================================
    //! \brief Unique config id for property of 'timeout in milliseconds'.
    //------------------------------------------------------------------------------
    static MICROVISION_SDK_API const std::string timeoutInMsConfigId;

    //==============================================================================
    //! \brief Unique config id for property of 'buffer size'.
    //------------------------------------------------------------------------------
    static MICROVISION_SDK_API const std::string bufferSizeConfigId;

    //========================================
    //! \brief Default timeout in milliseconds.
    //----------------------------------------
    static constexpr uint32_t defaultTimeoutInMsConstant{500U};

public:
    //========================================
    //! \brief Empty constructor.
    //----------------------------------------
    NetworkConfiguration();

    //========================================
    //! \brief Construct and update network properties with optional default values.
    //! \param[in] defaultLocalIpAddress        Default value for local IP address.
    //! \param[in] defaultLocalPort             (Optional) Default value for local port, default is \c 0.
    //! \param[in] defaultRemoteIpAddress       (Optional) Default value for remote IP address, default is \c nullptr.
    //! \param[in] defaultRemotePort            (Optional) Default value for remote port, default is \c 0.
    //! \param[in] defaultTimeoutInMs           (Optional) Default value for timeout in milliseconds, default is \c 500U.
    //! \param[in] defaultBufferSize            (Optional) Default value for buffer size, default is \c 0.
    //----------------------------------------
    NetworkConfiguration(const IpAddressPtr defaultLocalIpAddress,
                         const uint16_t defaultLocalPort           = 0U,
                         const IpAddressPtr defaultRemoteIpAddress = nullptr,
                         const uint16_t defaultRemotePort          = 0U,
                         const uint32_t defaultTimeoutInMs         = defaultTimeoutInMsConstant,
                         const uint32_t defaultBufferSize          = 0U);

    //========================================
    //! \brief Copy constructor to copy and update network properties.
    //! \param[in] other  Other NetworkConfiguration to copy.
    //----------------------------------------
    NetworkConfiguration(const NetworkConfiguration& other);

    //========================================
    //! \brief Disabled move constructor to ensure thread-safety.
    //----------------------------------------
    NetworkConfiguration(NetworkConfiguration&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~NetworkConfiguration() override;

public:
    //========================================
    //! \brief Validate package source would be acceptable by configuration.
    //! \note Will call acceptPackage(const NetworkDataPackage&)
    //!       if \c dataPackage is instance of NetworkDataPackage.
    //! \param[in] dataPackage  DataPackage to be validate.
    //! \returns Either \c true if data package is acceptable, otherwise \c false.
    //----------------------------------------
    virtual bool acceptPackage(const DataPackage& dataPackage) const;

public:
    //========================================
    //! \brief Get local IP address configuration property.
    //!
    //! To address the endpoint on the local machine.
    //!
    //! \note Will used mostly for udp network connections.
    //! \returns Local IP address configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<IpAddressPtr>& getLocalIpAddress();

    //========================================
    //! \brief Get local port configuration property.
    //!
    //! To address the endpoint on the local machine.
    //!
    //! \note Will used mostly for udp network connections.
    //! \returns Local port configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t>& getLocalPort();

    //========================================
    //! \brief Get remote IP address configuration property.
    //!
    //! To address the endpoint on the remote machine.
    //!
    //! \note Will used mostly for tcp network connections.
    //! \returns Remote IP address configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<IpAddressPtr>& getRemoteIpAddress();

    //========================================
    //! \brief Get remote port configuration property.
    //!
    //! To address the endpoint on the remote maschine.
    //!
    //! \note Will used mostly for tcp network connections.
    //! \returns Remote port configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t>& getRemotePort();

    //========================================
    //! \brief Get timeout in milliseconds configuration property.
    //! \returns Timeout in milliseconds configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint32_t>& getTimeoutInMs();

    //========================================
    //! \brief Get buffer size configuration property.
    //! \returns Buffer size configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint32_t>& getBufferSize();

protected:
    //========================================
    //! \brief Get string representation of the IP address.
    //! \param[in] ipAddress  Pointer to IP address
    //! \return The typical string representation of the IP address.
    //----------------------------------------
    std::string ipAddressToString(const IpAddressPtr& ipAddress) const;

    //========================================
    //! \brief Register value redirect if 'on set value' event is trigger.
    //! \param[in, out] ipPropertyOfAddress     Ip address property
    //! \param[in, out] ipPropertyOfString      Ipv4 string property
    //----------------------------------------
    void registerIpPropertySync(ConfigurationPropertyOfType<IpAddressPtr>& ipPropertyOfAddress,
                                ConfigurationPropertyOfType<std::string>& ipPropertyOfString);

    //========================================
    //! \brief Get UriProtocol of network configuration.
    //! \returns UriProtocol of network configuration.
    //----------------------------------------
    virtual UriProtocol getUriProtocol() const = 0;

    //========================================
    //! \brief Validate source uri would be acceptable by configuration.
    //! \param[in] source  Source Uri to be validated.
    //! \returns Either \c true if source Uri is acceptable, otherwise \c false.
    //----------------------------------------
    virtual bool acceptPackageSource(const Uri& source) const = 0;

    //========================================
    //! \brief Validate local and remote uri would be acceptable by configuration.
    //! \param[in] local   Local Uri to be validate.
    //! \param[in] remote  Remote Uri to be validate.
    //! \returns Either \c true if local and remote Uri is acceptable, otherwise \c false.
    //----------------------------------------
    virtual bool acceptPackageSource(const Uri& local, const Uri& remote) const = 0;

    //========================================
    //! \brief Validate address uri would be acceptable by IP and port configuration property.
    //! \param[in] ipProperty           Ip configuration property.
    //! \param[in] portProperty         Port configuration property.
    //! \param[in] address              Uri to be validate.
    //! \param[in] acceptAllIfUnset     (Optional) Accept all addresses if properties are not set, default \c false.
    //! \returns Either \c true if address Uri is acceptable, otherwise \c false.
    //----------------------------------------
    virtual bool acceptAddress(const ConfigurationPropertyOfType<IpAddressPtr>& ipProperty,
                               const ConfigurationPropertyOfType<uint16_t>& portProperty,
                               const Uri& address,
                               const bool acceptAllIfUnset = false) const;

protected:
    //========================================
    //! \brief Local IP address configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<IpAddressPtr> m_localIpAddress;

    //========================================
    //! \brief Local port configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t> m_localPort;

    //========================================
    //! \brief Remote IP address configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<IpAddressPtr> m_remoteIpAddress;

    //========================================
    //! \brief Remote port configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t> m_remotePort;

    //========================================
    //! \brief Timeout in milliseconds configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint32_t> m_timeoutInMs;

    //========================================
    //! \brief Buffer size configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint32_t> m_bufferSize;
}; // class NetworkConfiguration

//==============================================================================
//! \brief Nullable NetworkConfiguration pointer.
//------------------------------------------------------------------------------
using NetworkConfigurationPtr = std::shared_ptr<NetworkConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
