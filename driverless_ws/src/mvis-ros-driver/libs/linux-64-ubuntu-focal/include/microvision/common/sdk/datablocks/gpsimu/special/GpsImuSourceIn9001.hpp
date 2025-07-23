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
//! \date Feb 4, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {

//========================================
//!\brief Sources that provide GPS/IMU measurements.
//----------------------------------------
enum class GpsImuSourceIn9001 : uint8_t
{
    Can                 = 0,
    XSensImu            = 1,
    ThirdPartyOGpsImuRt = 2,
    GenesysAdma         = 3,
    SpatialDual         = 4,
    Tfc                 = 5,
    VBox3i              = 6,
    //  7 used
    //  8 used
    //  9 used
    // 10 used
    NovaGpsImu = 11,
    // 50 used
    Unknown = 99
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
