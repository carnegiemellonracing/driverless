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
//! \date Oct 25, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <string.h>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//! Flags representing the rotation-execution order reg. the sensors mounting
//! point transformation from sensor- to vehicle coord. system
enum class RotationOrder : uint8_t
{
    RollPitchYaw = 0x00,
    RollYawPitch = 0x01,
    PitchRollYaw = 0x02,
    PitchYawRoll = 0x03,
    YawRollPitch = 0x04,
    YawPitchRoll = 0x05
}; // RotationOrder

//==============================================================================

//==============================================================================
//! \brief Stream operator for writing the rotation order to a stream
//! \param[in] os             The stream, the rotation order shall be written to
//! \param[in] rotationOrder  The rotation order which shall be streamed
//! \return The stream to which the rotation order was written to
//------------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream& os, const RotationOrder rotationOrder)
{
    switch (rotationOrder)
    {
    case RotationOrder::RollPitchYaw:
        os << "RollPitchYaw";
        break;
    case RotationOrder::RollYawPitch:
        os << "RollYawPitch";
        break;
    case RotationOrder::PitchRollYaw:
        os << "PitchRollYaw";
        break;
    case RotationOrder::PitchYawRoll:
        os << "PitchYawRoll";
        break;
    case RotationOrder::YawRollPitch:
        os << "YawRollPitch";
        break;
    case RotationOrder::YawPitchRoll:
        os << "YawPitchRoll";
        break;

    default:
        throw std::invalid_argument("Unknown rotation order.");
    }

    return os;
}

//==============================================================================
//! \brief Stream operator for reading the rotation order from a stream
//! \param[in] is             The stream, the rotation order shall be written to
//! \param[in] rotationOrder  The rotation order which shall be streamed
//! \return The stream to which the rotation order was written to
//!
//! \note If reading the data failed (check with /a istream::fail()) the content of the rotationOrder is undefined.
//------------------------------------------------------------------------------
inline std::istream& operator>>(std::istream& is, RotationOrder& rotationOrder)
{
    char buffer[16] = {0};
    is.get(buffer, sizeof(buffer));
    if (is.fail())
    {
        // Error, give up.
        return is;
    }

    if (strcmp(buffer, "RollPitchYaw") == 0)
    {
        rotationOrder = RotationOrder::RollPitchYaw;
    }
    else if (strcmp(buffer, "RollYawPitch") == 0)
    {
        rotationOrder = RotationOrder::RollYawPitch;
    }
    else if (strcmp(buffer, "PitchRollYaw") == 0)
    {
        rotationOrder = RotationOrder::PitchRollYaw;
    }
    else if (strcmp(buffer, "PitchYawRoll") == 0)
    {
        rotationOrder = RotationOrder::PitchYawRoll;
    }
    else if (strcmp(buffer, "YawRollPitch") == 0)
    {
        rotationOrder = RotationOrder::YawRollPitch;
    }
    else if (strcmp(buffer, "YawPitchRoll") == 0)
    {
        rotationOrder = RotationOrder::YawPitchRoll;
    }
    else
    {
        // Unknown string -> error.
        is.setstate(std::ios::failbit);
    }

    return is;
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
