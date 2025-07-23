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

#include <microvision/common/sdk/config/devices/DeviceConfiguration.hpp>
#include <microvision/common/sdk/datablocks/MountingPosition.hpp>
#include <microvision/common/sdk/RotationOrder.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Abstract base for Sensor device configurations.
//!
//! \extends microvision::common::sdk::DeviceConfiguration
//------------------------------------------------------------------------------
class SensorConfiguration : public DeviceConfiguration
{
public:
    //==============================================================================
    //! \brief Unique config id for property of 'mounting postion'.
    //------------------------------------------------------------------------------
    static const MICROVISION_SDK_API std::string mountingPositionConfigId;

    //==============================================================================
    //! \brief Unique config id for property of 'rotation order'.
    //------------------------------------------------------------------------------
    static const MICROVISION_SDK_API std::string rotationOrderConfigId;

    //==============================================================================
    //! \brief Unique config id for property of 'rotation position offset'.
    //------------------------------------------------------------------------------
    static const MICROVISION_SDK_API std::string rotationalPositionOffsetConfigId;

public:
    //========================================
    //! \brief Construct and update sensor properties.
    //----------------------------------------
    SensorConfiguration();

    //========================================
    //! \brief Copy constructor to copy and update sensor properties.
    //! \param[in] other  Other SensorConfiguration to copy.
    //----------------------------------------
    SensorConfiguration(const SensorConfiguration& other);

    //========================================
    //! \brief Disabled move constructor because of thread-safe guarantee.
    //----------------------------------------
    SensorConfiguration(SensorConfiguration&&) = delete;

    //========================================
    //! \brief Default destructor.
    //----------------------------------------
    ~SensorConfiguration() override;

public:
    //========================================
    //! \brief Get mounting position configuration property.
    //! \returns Mounting position configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<MountingPosition<float>>& getMountingPosition();

    //========================================
    //! \brief Get rotation order configuration property.
    //! \returns Rotation order configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<RotationOrder>& getRotationOrder();

    //========================================
    //! \brief Get rotational position offset configuration property.
    //! \returns Rotational position offset configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<float>& getRotationalPositionOffset();

private:
    //========================================
    //! \brief Mounting position configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<MountingPosition<float>> m_mountingPosition;

    //========================================
    //! \brief Rotation order configuration property.
    //----------------------------------------
    ConfigurationPropertyOfType<RotationOrder> m_rotationOrder;

    //========================================
    //! \brief Rotational position offset.
    //----------------------------------------
    ConfigurationPropertyOfType<float> m_rotationalPositionOffset;

}; // class SensorConfiguration

//==============================================================================
//! \brief Nullable SensorConfiguration pointer.
//------------------------------------------------------------------------------
using SensorConfigurationPtr = std::shared_ptr<SensorConfiguration>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
