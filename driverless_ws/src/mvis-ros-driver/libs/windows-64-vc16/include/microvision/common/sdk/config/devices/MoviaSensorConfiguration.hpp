//==============================================================================
//! \file
//!
//! \brief Configuration for the prototype MOVIA B0 sensor device.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 28, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/io/UdpConfiguration.hpp>
#include <microvision/common/sdk/config/EnumConfigurationPropertyOf.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Enumeration list of transport protocol chains.
//------------------------------------------------------------------------------
enum class MoviaTransportProtocol : uint8_t
{
    Iutp = 0, //!< Transport protocol chain UDP -> IUTP -> INTP.
    Intp = 1, //!< Transport protocol chain UDP -> INTP.
};

//==============================================================================
//! \brief Configuration for the prototype MOVIA B0 sensor device.
//!
//! \note Please note that using recent MOVIA sensors require the movia-device-plugin to be loaded!
//!
//! Example MOVIA device configuration:
//! \code
//! auto deviceConfig = ConfigurationFactory::getInstance().createConfiguration(MoviaSensorConfiguration::typeName);
//! deviceConfig->trySetValue("multicast_ip", makeIp("239.1.2.5")); // if false: configuration property does not exists or type is incompatible!
//! deviceConfig->trySetValue("local_port", uint16_t{12349}); // if false: configuration property does not exists or type is incompatible!
//!
//! // Following the different ways to set the transport protocol:
//! deviceConfig->trySetValue("transport_protocol", MoviaTransportProtocol::Iutp); // if false: configuration property does not exists or type is incompatible!
//! deviceConfig->trySetValue("transport_protocol", "Intp"); // if false: configuration property does not exists or type is incompatible!
//! deviceConfig->trySetValue("transport_protocol", uint8_t{0U}); // if false: configuration property does not exists or type is incompatible!
//!
//! device->setDeviceConfiguration(deviceConfig); // if false: device configuration failed
//! \endcode
//!
//! New configuration properties added:
//! Property Name       | Type                                         | Description                                                                            | Default
//! ------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------
//! transport_protocol  | enum of MoviaTransportProtocol | Transfer protocol chain which has to be used for MOVIA device.  | MoviaTransportProtocol::Iutp
//!
//! \sa UdpConfiguration
//------------------------------------------------------------------------------
class MoviaSensorConfiguration : public UdpConfiguration
{
public:
    //========================================
    //! \brief Configuration type name
    //----------------------------------------
    static MICROVISION_SDK_API const std::string typeName;

    //========================================
    //! \brief Unique config id for property 'transport_protocol'
    //----------------------------------------
    static MICROVISION_SDK_API const std::string transportProtocolConfigId;

public:
    //========================================
    //! \brief Get name of type of this configuration.
    //! \returns Configuration type.
    //----------------------------------------
    static MICROVISION_SDK_API const std::string& getTypeName();

public:
    //========================================
    //! \brief Default construct
    //----------------------------------------
    MoviaSensorConfiguration(const MoviaTransportProtocol defaultTransportProtocol = MoviaTransportProtocol::Iutp);

    //========================================
    //! \brief Copy constructor
    //----------------------------------------
    MoviaSensorConfiguration(const MoviaSensorConfiguration& other);

    //========================================
    //! \brief Disable move constructor to ensure thread safety
    //----------------------------------------
    MoviaSensorConfiguration(UdpConfiguration&&) = delete;

    //========================================
    //! \brief Destructor
    //----------------------------------------
    ~MoviaSensorConfiguration() override;

public:
    //========================================
    //! \brief Return the configuration type
    //! \returns Configuration type
    //----------------------------------------
    const std::string& getType() const override;

    //========================================
    //! \brief Get copy of configuration
    //! \returns Pointer to newly copied configuration
    //----------------------------------------
    ConfigurationPtr copy() const override;

public:
    //========================================
    //! \brief Get property of transport protocol chain to receive data from sensor.
    //! \returns Property of transport protocol chain.
    //----------------------------------------
    EnumConfigurationPropertyOfType<MoviaTransportProtocol>& getTransportProtocol();

private:
    //========================================
    //! \brief Property for transport protocol chain.
    //----------------------------------------
    EnumConfigurationPropertyOfType<MoviaTransportProtocol> m_transportProtocol;
}; // class MoviaSensorConfiguration

//=================================================
//! \brief Nullable MoviaSensorConfiguration pointer
//-------------------------------------------------
using MoviaSensorConfigurationPtr = std::shared_ptr<MoviaSensorConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
