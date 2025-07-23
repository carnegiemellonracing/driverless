//==============================================================================
//! \file
//!
//! \brief MVIS ECU UDP device configuration.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 18, 2023
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/config/devices/DeviceConfiguration.hpp>
#include <microvision/common/sdk/config/io/UdpConfiguration.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Configuration for the MVIS ECU UDP device.
//!
//! Example UDP ECU device configuration:
//! \code
//! auto deviceConfig = ConfigurationFactory::getInstance().createConfiguration(MvisEcuUdpConfiguration::typeName);
//! deviceConfig->trySetValue("multicast_ip", makeIp("239.1.2.5")); // if false: configuration property does not exists or type is incompatible!
//! deviceConfig->trySetValue("local_port", uint16_t{12349}); // if false: configuration property does not exists or type is incompatible!
//!
//! device->setDeviceConfiguration(deviceConfig); // if false: device configuration failed
//! \endcode
//!
//! New configuration properties added:
//! Property Name      | Type        | Description                                                  | Default
//! ------------------ | ----------- | ------------------------------------------------------------ | -------------
//!
//! \sa UdpConfiguration
//------------------------------------------------------------------------------
class MvisEcuUdpConfiguration : public UdpConfiguration
{
public:
    //========================================
    //! \brief Configuration type name
    //----------------------------------------
    static MICROVISION_SDK_API const std::string typeName;

public:
    //========================================
    //! \brief Default constructor.
    //----------------------------------------
    MvisEcuUdpConfiguration();

    //========================================
    //! \brief Copy constructor.
    //----------------------------------------
    MvisEcuUdpConfiguration(const MvisEcuUdpConfiguration& other);

    //========================================
    //! \brief Disable move constructor to ensure thread safety
    //----------------------------------------
    MvisEcuUdpConfiguration(MvisEcuUdpConfiguration&&) = delete;

    //========================================
    //! \brief Destructor
    //----------------------------------------
    ~MvisEcuUdpConfiguration() override = default;

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
};

//=================================================
//! \brief Nullable MvisEcuUdpConfiguration pointer
//-------------------------------------------------
using MvisEcuUdpConfigurationPtr = std::shared_ptr<MvisEcuUdpConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
