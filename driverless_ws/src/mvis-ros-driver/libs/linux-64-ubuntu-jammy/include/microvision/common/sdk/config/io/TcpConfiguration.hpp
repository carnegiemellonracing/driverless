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
//! \date Feb 21, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/io/NetworkConfiguration.hpp>
#include <microvision/common/sdk/config/EnumConfigurationPropertyOf.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Tcp network configuration.
//!
//! This configuration contains all the required properties for an tcp connection.
//!
//! \extends microvision::common::sdk::NetworkConfiguration
//------------------------------------------------------------------------------
class TcpConfiguration : public NetworkConfiguration
{
public:
    //========================================
    //! \brief Configuration type name.
    //----------------------------------------
    static const MICROVISION_SDK_API std::string typeName;

    //==============================================================================
    //! \brief Unique config id for property of 'reconnect mode'.
    //------------------------------------------------------------------------------
    static const MICROVISION_SDK_API std::string reconnectModeConfigId;

    //==============================================================================
    //! \brief Unique config id for property of 'reconnect timeout in ms'.
    //------------------------------------------------------------------------------
    static const MICROVISION_SDK_API std::string reconnectTimeoutInMsConfigId;

    //========================================
    //! \brief Default reconnect timeout in milliseconds.
    //----------------------------------------
    static constexpr uint32_t defaultReconnectTimeoutInMsConstant{500U};

public:
    //========================================
    //! \brief Reconnection modes.
    //----------------------------------------
    enum class ReconnectMode : uint8_t
    {
        //========================================
        //! \brief Immediately connect to the remote device.
        //!
        //! If failed or if the connection gets lost afterwards, it will not be re-established.
        //----------------------------------------
        InitialConnect = 0,

        //========================================
        //! \brief Try to connect until the remote device is available.
        //!
        //! If the connection gets lost afterwards, it will not be re-established.
        //----------------------------------------
        WaitForRemoteDevice = 1,

        //========================================
        //! \brief Try to connect until the remote device is available.
        //!
        //! Reconnect every time the remote device is disconnected.
        //----------------------------------------
        AutoReconnect = 2
    };

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
    TcpConfiguration();

    //========================================
    //! \brief Construct and update tcp network properties with optional default values.
    //! \param[in] defaultLocalIpAddress        Default value for local IP address.
    //! \param[in] defaultLocalPort             (Optional) Default value for local port, default is \c 0.
    //! \param[in] defaultRemoteIpAddress       (Optional) Default value for remote IP address, default is \c "".
    //! \param[in] defaultRemotePort            (Optional) Default value for remote port, default is \c 0.
    //! \param[in] defaultTimeoutInMs           (Optional) Default value for timeout in milliseconds, default is \c 500U.
    //! \param[in] defaultBufferSize            (Optional) Default value for buffer size, default is \c 0.
    //! \param[in] defaultReconnectMode         (Optional) Default value for reconnect mode, default is \c InitialConnect.
    //! \param[in] defaultReconnectTimeoutInMs  (Optional) Default value for reconnect timeout in ms, default is \c 500U.
    //----------------------------------------
    TcpConfiguration(const IpAddressPtr defaultLocalIpAddress,
                     const uint16_t defaultLocalPort            = 0U,
                     const IpAddressPtr defaultRemoteIpAddress  = nullptr,
                     const uint16_t defaultRemotePort           = 0U,
                     const uint32_t defaultTimeoutInMs          = defaultTimeoutInMsConstant,
                     const uint32_t defaultBufferSize           = 0U,
                     const ReconnectMode defaultReconnectMode   = ReconnectMode::InitialConnect,
                     const uint32_t defaultReconnectTimeoutInMs = defaultReconnectTimeoutInMsConstant);

    //========================================
    //! \brief Copy constructor to copy and update tcp network properties.
    //! \param[in] other  Other TcpConfiguration to copy.
    //----------------------------------------
    TcpConfiguration(const TcpConfiguration&);

    //========================================
    //! \brief Disabled move constructor because of thread-safe guarantee.
    //----------------------------------------
    TcpConfiguration(TcpConfiguration&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~TcpConfiguration() override;

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
    //! \brief Get reconnect mode configuration property.
    //! \returns Reconnect mode configuration property.
    //----------------------------------------
    EnumConfigurationPropertyOfType<ReconnectMode>& getReconnectMode();

    //========================================
    //! \brief Get reconnect timeout in milliseconds configuration property.
    //! \returns Reconnect timeout in milliseconds configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint32_t>& getReconnectTimeoutInMs();

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
    //! \brief Reconnect mode configuration property.
    //----------------------------------------
    EnumConfigurationPropertyOfType<ReconnectMode> m_reconnectMode;

    //========================================
    //! \brief Reconnect timeout in milliseconds configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint32_t> m_reconnectTimeoutInMs;

}; // class UdpConfiguration

//==============================================================================
//! \brief Nullable UdpConfiguration pointer.
//------------------------------------------------------------------------------
using TcpConfigurationPtr = std::shared_ptr<TcpConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
