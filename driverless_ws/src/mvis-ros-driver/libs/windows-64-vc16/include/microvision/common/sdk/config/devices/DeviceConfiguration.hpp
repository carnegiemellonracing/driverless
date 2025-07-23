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

#include <microvision/common/sdk/config/ConfigurationPropertyOfType.hpp>
#include <microvision/common/sdk/config/Configuration.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Abstract base for device configuration.
//!
//! \extends microvision::common::sdk::Configuration
//------------------------------------------------------------------------------
class DeviceConfiguration : public virtual Configuration
{
public:
    //==============================================================================
    //! \brief Unique config id for property of 'device id'.
    //------------------------------------------------------------------------------
    static MICROVISION_SDK_API const std::string deviceIdConfigId;

    //========================================
    //! \brief Device id for unknown is \c 0.
    //----------------------------------------
    static constexpr uint8_t deviceIdForUnknown{0x0U};

    //========================================
    //! \brief Device id for any is \c 255.
    //----------------------------------------
    static constexpr uint8_t deviceIdForAny{0xFFU};

public:
    //========================================
    //! \brief Construct and update device properties.
    //----------------------------------------
    DeviceConfiguration();

    //========================================
    //! \brief Copy constructor to copy and update device properties.
    //! \param[in] other  Other DeviceConfiguration to copy.
    //----------------------------------------
    DeviceConfiguration(const DeviceConfiguration& other);

    //========================================
    //! \brief Disabled move constructor to ensure thread-safety.
    //----------------------------------------
    DeviceConfiguration(DeviceConfiguration&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~DeviceConfiguration() override;

public:
    //========================================
    //! \brief Get device id configuration property.
    //! \returns Device id configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint8_t>& getDeviceId();

private:
    //========================================
    //! \brief Buffer size configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<uint8_t> m_deviceId;
}; // class DeviceConfiguration

//==============================================================================
//! \brief Nullable DeviceConfiguration pointer.
//------------------------------------------------------------------------------
using DeviceConfigurationPtr = std::shared_ptr<DeviceConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
