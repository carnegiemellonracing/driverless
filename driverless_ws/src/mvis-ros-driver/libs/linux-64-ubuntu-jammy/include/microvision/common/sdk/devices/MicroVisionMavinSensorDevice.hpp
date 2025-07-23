//==============================================================================
//! \file
//!
//! \brief Mavin sensor device setup function definition.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jan 23, 2023
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/devices/IdcDevice.hpp>

#include <microvision/common/sdk/datablocks/MountingPosition.hpp>
#include <microvision/common/sdk/datablocks/mavinRaw/special/MavinRawSensorInfoIn2360.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Helper function to easily create a MAVIN device
//! \param[in] ip                Remote IP address for connection.
//! \param[in] deviceId          Device id to use for this device. Set in the scan data.
//! \param[in] mountingPosition  Mounting position to be used for the pointcloud data. Contains X, Y, Z, Yaw, Pitch, Roll.
//! \param[in] format            Determines which TCP stream format is expected from the MAVIN sensor.
//! \param[in] port              Remote port for connection. Normally 27000 for MVO, 17000 for PCRAW format.
//! \returns  Mavin device which is configured according to input parameters.
//------------------------------------------------------------------------------
IdcDevicePtr createMavinDevice(const std::string& ip,
                               const uint8_t deviceId                          = 1,
                               const MountingPosition<float>& mountingPosition = {},
                               const MavinDataFormat format                    = MavinDataFormat::MVO,
                               const uint16_t port                             = 27000);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
