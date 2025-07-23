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
//!\brief Quality flags for the GPS/IMU measurements.
//----------------------------------------
enum class GpsImuInsQualityIn9001 : uint8_t
{
    Gps            = 0, //!< Only GPS (default). filtered output
    DGpsCourse     = 1, //!< Pre-RTK. filtered output
    DGpsPrecise    = 2, //!< RTK precision. filtered output
    NoFix          = 3, //!< No GPS-Fix, Predicted position only
    GpsRaw         = 4, //!< Only GPS, raw, unfiltered output
    DGpsCourseRaw  = 5, //!< Pre-RTK, raw, unfiltered output
    DGpsPreciseRaw = 6, //!< RTK precision, unfiltered output
    NotReliable    = 7 //!< Sensor output is not reliable, sensor internal filter might not be initialized
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
