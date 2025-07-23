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
namespace velocity {
//==============================================================================

//==============================================================================
// velocity constants
// available and convertible velocity units
//------------------------------------------------------------------------------

static constexpr uint64_t meterPerSecond   = common::sdk::hash("meterpersecond");
static constexpr uint64_t kilometerPerHour = common::sdk::hash("kilometerperhour");
static constexpr uint64_t milesPerHour     = common::sdk::hash("milesperhour");

//==============================================================================
} // namespace velocity
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

//==============================================================================
// names and symbols
//------------------------------------------------------------------------------
template<>
struct Unit<velocity::meterPerSecond>
{
    constexpr static const char* name() { return "meterPerSecond"; }
    constexpr static const char* symbol() { return "m/s"; }
};
template<>
struct Unit<velocity::kilometerPerHour>
{
    constexpr static const char* name() { return "kilometerPerHour"; }
    constexpr static const char* symbol() { return "km/h"; }
};
template<>
struct Unit<velocity::milesPerHour>
{
    constexpr static const char* name() { return "milesPerHour"; }
    constexpr static const char* symbol() { return "mph"; }
};

//==============================================================================

//==============================================================================
// conversions
//------------------------------------------------------------------------------
template<typename T>
struct Convert<velocity::kilometerPerHour, velocity::meterPerSecond, T>
{
    constexpr T operator()(const T& valInKilometerPerHour) const
    {
        return static_cast<T>((valInKilometerPerHour * static_cast<T>(length::metersPerKilometer))
                              / static_cast<T>(time::secondsPerHour));
    }
};
template<typename T>
struct Convert<velocity::meterPerSecond, velocity::kilometerPerHour, T>
{
    constexpr T operator()(const T& val) const
    {
        static_assert(std::numeric_limits<T>::max() > time::secondsPerHour, "Used type too small for calculation!");
        static_assert(std::numeric_limits<T>::max() > length::metersPerKilometer,
                      "Used type too small for calculation!");
        return static_cast<T>((val * static_cast<T>(time::secondsPerHour))
                              / static_cast<T>(length::metersPerKilometer));
    }
};
template<typename T>
struct Convert<velocity::milesPerHour, velocity::kilometerPerHour, T>
{
    constexpr T operator()(const T& valInMilesPerHour) const
    {
        return Convert<length::mile, length::kilometer, T>()(valInMilesPerHour);
    }
};
template<typename T>
struct Convert<velocity::kilometerPerHour, velocity::milesPerHour, T>
{
    constexpr T operator()(const T& valInKilometerPerHour) const
    {
        return Convert<length::kilometer, length::mile, T>()(valInKilometerPerHour);
    }
};

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
