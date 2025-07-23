//==============================================================================
//! \file
//!
//! \brief Configuration for the rtsp stream device.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 20, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/devices/DeviceConfiguration.hpp>
#include <microvision/common/sdk/config/io/TcpConfiguration.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Configuration for the rtsp stream device.
//!
//! Example rtsp stream device configuration:
//! \code
//! auto deviceConfig = ConfigurationFactory::getInstance().createConfiguration(RtspImageStreamConfiguration::typeName);
//! deviceConfig->trySetValue("device_id", uint8_t{1}); // if false: configuration property does not exists or type is incompatible!
//! deviceConfig->trySetValue("remote_ip", makeIp("192.168.172.5")); // if false: configuration property does not exists or type is incompatible!
//! deviceConfig->trySetValue("remote_port", uint16_t{12345}); // if false: configuration property does not exists or type is incompatible!
//! deviceConfig->trySetValue("path_of_url", std::string{"/cgi-bin/jpeg"}); // if false: configuration property does not exists or type is incompatible!
//! deviceConfig->trySetValue("rtp_port", uint16_t{30000}); // if false: configuration property does not exists or type is incompatible!
//! deviceConfig->trySetValue("rtcp_port", uint16_t{30001}); // if false: configuration property does not exists or type is incompatible!
//!
//! device->setDeviceConfiguration(deviceConfig); // if false: device configuration failed
//! \endcode
//!
//! New configuration properties added:
//! Property Name   | Type      | Description   | Default
//! --------------- | --------- | ------------- | -----------
//! path_of_url     | string    | Path of URL   | ""
//! rtp_port        | uint16_t  | RTP port      | 50000
//! rtcp_port       | uint16_t  | RTCP port     | 50001
//!
//! \sa TcpConfiguration
//------------------------------------------------------------------------------
class RtspImageStreamConfiguration : public DeviceConfiguration, public TcpConfiguration
{
public:
    //========================================
    //! \brief Configuration type name
    //----------------------------------------
    static MICROVISION_SDK_API const std::string typeName;

    //========================================
    //! \brief Unique config id for property 'path_of_url'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string pathOfUrlConfigId;

    //========================================
    //! \brief Unique config id for property 'rtp_port'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string rtpPortConfigId;

    //========================================
    //! \brief Unique config id for property 'rtcp_port'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string rtcpPortConfigId;

    //========================================
    //! \brief Default RTSP port.
    //----------------------------------------
    static MICROVISION_SDK_API constexpr uint16_t defaultRtspPort{554U};

public:
    //========================================
    //! \brief Get name of type of this configuration.
    //! \returns Configuration type.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string& getTypeName();

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    RtspImageStreamConfiguration(const std::string& defaultPathOfUrl = "",
                                 const uint16_t& defaultRtpPort      = 50000U,
                                 const uint16_t& defaultRtcpPort     = 50001U);

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    RtspImageStreamConfiguration(const RtspImageStreamConfiguration& other);

    //========================================
    //! \brief Disable move constructor to ensure thread safety.
    //----------------------------------------
    RtspImageStreamConfiguration(RtspImageStreamConfiguration&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~RtspImageStreamConfiguration() override;

public:
    //========================================
    //! \brief Return the configuration type.
    //! \returns Configuration type.
    //----------------------------------------
    const std::string& getType() const override;

    //========================================
    //! \brief Get copy of configuration.
    //! \returns Pointer to newly copied configuration.
    //----------------------------------------
    ConfigurationPtr copy() const override;

public:
    //========================================
    //! \brief Get property of url path.
    //! \returns Property of url path.
    //----------------------------------------
    ConfigurationPropertyOfType<std::string>& getPathOfUrl();

    //========================================
    //! \brief Get property of rtp port.
    //! \returns Property of rtp port.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t>& getRtpPort();

    //========================================
    //! \brief Get property of rtcp port.
    //! \returns Property of rtcp port.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t>& getRtcpPort();

private:
    //========================================
    //! \brief Property for path of url.
    //----------------------------------------
    ConfigurationPropertyOfType<std::string> m_pathOfUrl;

    //========================================
    //! \brief Property of rtp port.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t> m_rtpPort;

    //========================================
    //! \brief Property of rtcp port.
    //----------------------------------------
    ConfigurationPropertyOfType<uint16_t> m_rtcpPort;

}; // class RtspImageStreamConfiguration

//=================================================
//! \brief Nullable RtspImageStreamConfiguration pointer.
//-------------------------------------------------
using RtspImageStreamConfigurationPtr = std::shared_ptr<RtspImageStreamConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
