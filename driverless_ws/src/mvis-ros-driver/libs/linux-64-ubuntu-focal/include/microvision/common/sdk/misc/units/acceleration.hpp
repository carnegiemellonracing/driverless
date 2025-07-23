//==============================================================================
//! \file
//!
//! \brief Contains helper functions for converting to SI units.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 25, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/misc/units/Unit.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace unit {
namespace acceleration {
//==============================================================================

// acceleration constants
constexpr double gravity = 9.80665;

// available and convertible acceleration units
static constexpr uint64_t meterPerSecond2 = common::sdk::hash("meterpersecond2");
static constexpr uint64_t footPerSecond2  = common::sdk::hash("footpersecond2");

//==============================================================================
} // namespace acceleration
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace unit {
//==============================================================================

// names and symbols
template<>
struct Unit<acceleration::meterPerSecond2>
{
    constexpr static const char* name() { return "meterPerSecond2"; }
    constexpr static const char* symbol() { return "m/s^2"; }
};

template<>
struct Unit<acceleration::footPerSecond2>
{
    constexpr static const char* name() { return "footPerSecond2"; }
    constexpr static const char* symbol() { return "f/s^2"; }
};

//==============================================================================

// conversions
template<typename T>
struct Convert<acceleration::meterPerSecond2, acceleration::footPerSecond2, T>
{
    constexpr T operator()(const T& val) const { return Convert<length::meter, length::foot, T>()(val); }
};
template<typename T>
struct Convert<acceleration::footPerSecond2, acceleration::meterPerSecond2, T>
{
    constexpr T operator()(const T& valInFootPerSecond2) const
    {
        return Convert<length::foot, length::meter, T>(valInFootPerSecond2);
    }
};

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
