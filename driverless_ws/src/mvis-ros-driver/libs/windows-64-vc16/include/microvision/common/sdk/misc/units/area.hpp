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
namespace area {
//==============================================================================

// area constants
static constexpr uint32_t kilometer2PerMeter2       = unit::kilo * unit::kilo;
static constexpr uint32_t hectaresPerKilometer2     = 10 * 10;
static constexpr uint32_t meter2PerHectare          = 100 * 100;
static constexpr uint32_t feet2PerMeter2Numerator   = length::feetPerMeterNumerator * length::feetPerMeterNumerator;
static constexpr uint32_t feet2PerMeter2Denominator = length::feetPerMeterDenominator * length::feetPerMeterDenominator;
static constexpr uint32_t feet2PerMile2             = length::feetPerMile * length::feetPerMile;
static constexpr uint32_t feet2PerYard2             = length::feetPerYard * length::feetPerYard;

// available and convertible area units
static constexpr uint64_t meter2     = common::sdk::hash("meter2");
static constexpr uint64_t kilometer2 = common::sdk::hash("kilometer2");
static constexpr uint64_t hectare    = common::sdk::hash("hectare");
static constexpr uint64_t foot2      = common::sdk::hash("foot2");
static constexpr uint64_t acre       = common::sdk::hash("acre");
static constexpr uint64_t mile2      = common::sdk::hash("mile2");

//==============================================================================
} // namespace area
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
struct Unit<area::meter2>
{
    constexpr static const char* name() { return "meter2"; }
    constexpr static const char* symbol() { return "m^2"; }
};
template<>
struct Unit<area::kilometer2>
{
    constexpr static const char* name() { return "kilometer2"; }
    constexpr static const char* symbol() { return "km^2"; }
};
template<>
struct Unit<area::hectare>
{
    constexpr static const char* name() { return "hectare"; }
    constexpr static const char* symbol() { return "ha"; }
};
template<>
struct Unit<area::foot2>
{
    constexpr static const char* name() { return "foot2"; }
    constexpr static const char* symbol() { return "ft^2"; }
};
template<>
struct Unit<area::acre>
{
    constexpr static const char* name() { return "acre"; }
    constexpr static const char* symbol() { return "acre"; }
};
template<>
struct Unit<area::mile2>
{
    constexpr static const char* name() { return "mile2"; }
    constexpr static const char* symbol() { return "mile^2"; }
};

//==============================================================================

// conversions
template<typename T>
struct Convert<area::kilometer2, area::meter2, T>
{
    constexpr T operator()(const T& valInKilometer2) const
    {
        return static_cast<T>(valInKilometer2 * static_cast<T>(area::kilometer2PerMeter2));
    }
};
template<typename T>
struct Convert<area::meter2, area::kilometer2, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val / static_cast<T>(area::kilometer2PerMeter2));
    }
};
template<typename T>
struct Convert<area::hectare, area::kilometer2, T>
{
    constexpr T operator()(const T& valInHectares) const
    {
        return static_cast<T>(valInHectares / static_cast<T>(area::hectaresPerKilometer2));
    }
};
template<typename T>
struct Convert<area::kilometer2, area::hectare, T>
{
    constexpr T operator()(const T& valInKilometer2) const
    {
        return static_cast<T>(valInKilometer2 * static_cast<T>(area::hectaresPerKilometer2));
    }
};
template<typename T>
struct Convert<area::hectare, area::meter2, T>
{
    constexpr T operator()(const T& valInHectares) const
    {
        return static_cast<T>(valInHectares * static_cast<T>(area::meter2PerHectare));
    }
};
template<typename T>
struct Convert<area::meter2, area::hectare, T>
{
    constexpr T operator()(const T& val) const { return static_cast<T>(val / static_cast<T>(area::meter2PerHectare)); }
};

template<typename T>
struct Convert<area::foot2, area::meter2, T>
{
    constexpr T operator()(const T& valInFoot2) const
    {
        return static_cast<T>((valInFoot2 * static_cast<T>(area::feet2PerMeter2Numerator))
                              / static_cast<T>(area::feet2PerMeter2Denominator));
    }
};
template<typename T>
struct Convert<area::meter2, area::foot2, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>((val * static_cast<T>(area::feet2PerMeter2Denominator))
                              / static_cast<T>(area::feet2PerMeter2Numerator));
    }
};
template<typename T>
struct Convert<area::acre, area::foot2, T>
{
    constexpr T operator()(const T& valInAcres) const
    {
        return static_cast<T>(valInAcres * static_cast<T>(area::feet2PerYard2));
    }
};
template<typename T>
struct Convert<area::foot2, area::acre, T>
{
    constexpr T operator()(const T& valInFoot2) const
    {
        return static_cast<T>(valInFoot2 / static_cast<T>(area::feet2PerYard2));
    }
};
template<typename T>
struct Convert<area::mile2, area::foot2, T>
{
    constexpr T operator()(const T& valInMile2) const
    {
        return static_cast<T>(valInMile2 * static_cast<T>(area::feet2PerMile2));
    }
};
template<typename T>
struct Convert<area::foot2, area::mile2, T>
{
    constexpr T operator()(const T& valInFoot2) const
    {
        return static_cast<T>(valInFoot2 / static_cast<T>(area::feet2PerMile2));
    }
};

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
