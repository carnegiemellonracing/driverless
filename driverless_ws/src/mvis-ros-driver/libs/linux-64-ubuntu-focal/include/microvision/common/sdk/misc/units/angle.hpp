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
//! \date Apr 24, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/misc/SdkHash.hpp>

#include <microvision/common/sdk/misc/units/Unit.hpp>
#include <microvision/common/sdk/Math.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace unit {
namespace angle {
//==============================================================================

// angle constants
static constexpr uint32_t degreesPerRotation          = 360;
static constexpr uint32_t degreesPerHalfRotation      = 180;
static constexpr uint32_t centiDegreesPerRotation     = degreesPerRotation * unit::centi;
static constexpr uint32_t centiDegreesPerHalfRotation = degreesPerHalfRotation * unit::centi;
static constexpr uint32_t tick32sPerFullRotation      = 11520;

// available and convertible angle units
static constexpr uint64_t radian      = common::sdk::hash("radian");
static constexpr uint64_t degree      = common::sdk::hash("degree");
static constexpr uint64_t centidegree = common::sdk::hash("centidegree");
static constexpr uint64_t arcminute   = common::sdk::hash("arcminute");
static constexpr uint64_t arcsecond   = common::sdk::hash("arcsecond");
static constexpr uint64_t tick32      = common::sdk::hash("tick32");

//==============================================================================
} // namespace angle
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
struct Unit<angle::radian>
{
    constexpr static const char* name() { return "radian"; }
    constexpr static const char* symbol() { return "rad"; }
};
template<>
struct Unit<angle::degree>
{
    constexpr static const char* name() { return "degree"; }
    constexpr static const char* symbol() { return "°"; }
};
template<>
struct Unit<angle::centidegree>
{
    constexpr static const char* name() { return "centidegree"; }
    constexpr static const char* symbol() { return ""; }
};
template<>
struct Unit<angle::arcminute>
{
    constexpr static const char* name() { return "arcminute"; }
    constexpr static const char* symbol() { return "'"; }
};
template<>
struct Unit<angle::arcsecond>
{
    constexpr static const char* name() { return "arcsecond"; }
    constexpr static const char* symbol() { return "\""; }
};
template<>
struct Unit<angle::tick32>
{
    constexpr static const char* name() { return "tick32"; }
    constexpr static const char* symbol() { return ""; }
};

//==============================================================================

// conversions
template<typename T>
struct Convert<angle::degree, angle::radian, T>
{
    constexpr T operator()(const T& valInDegrees) const
    {
        return static_cast<T>((valInDegrees * static_cast<T>(pi)) / static_cast<T>(angle::degreesPerHalfRotation));
    }
};

template<>
struct Convert<angle::radian, angle::degree, double>
{
    constexpr double operator()(const double& val) const { return val * rad2deg; }
};
template<>
struct Convert<angle::radian, angle::degree, float>
{
    constexpr float operator()(const float& val) const { return val * rad2degf; }
};
template<typename T>
struct Convert<angle::radian, angle::degree, T>
{
    static_assert(std::numeric_limits<T>::is_integer == false, "Integer type not usable for radian unit!");
    constexpr T operator()(const T& val) const
    {
        return (val * static_cast<T>(angle::degreesPerHalfRotation)) / static_cast<T>(pi);
    }
};

template<typename T>
struct Convert<angle::centidegree, angle::radian, T>
{
    constexpr T operator()(const T& valInCentidegrees) const
    {
        return Convert<angle::degree, angle::radian, T>()(valInCentidegrees / static_cast<T>(unit::centi));
    }
};

template<typename T>
struct Convert<angle::radian, angle::centidegree, T>
{
    static_assert(std::numeric_limits<T>::is_integer == false, "Integer type not usable for radian unit!");
    constexpr T operator()(const T& val) const
    {
        return Convert<angle::radian, angle::degree, T>()(val) * static_cast<T>(unit::centi);
    }
};

template<typename T>
struct Convert<angle::arcminute, angle::degree, T>
{
    constexpr T operator()(const T& valInArcMinutes) const
    {
        return static_cast<T>(valInArcMinutes / static_cast<T>(unit::time::minutesPerHour));
    }
};
template<typename T>
struct Convert<angle::degree, angle::arcminute, T>
{
    constexpr T operator()(const T& valInDegrees) const
    {
        return static_cast<T>(valInDegrees * static_cast<T>(unit::time::minutesPerHour));
    }
};
template<typename T>
struct Convert<angle::arcsecond, angle::arcminute, T>
{
    constexpr T operator()(const T& valInArcSeconds) const
    {
        return static_cast<T>(valInArcSeconds / static_cast<T>(microvision::common::sdk::unit::time::secondsPerMinute));
    }
};
template<typename T>
struct Convert<angle::arcminute, angle::arcsecond, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val * static_cast<T>(microvision::common::sdk::unit::time::secondsPerMinute));
    }
};
template<typename T>
struct Convert<angle::arcsecond, angle::degree, T>
{
    constexpr T operator()(const T& valInArcSeconds) const
    {
        return static_cast<T>(valInArcSeconds / static_cast<T>(microvision::common::sdk::unit::time::secondsPerHour));
    }
};
template<typename T>
struct Convert<angle::degree, angle::arcsecond, T>
{
    constexpr T operator()(const T& valInDegrees) const
    {
        return static_cast<T>(valInDegrees * static_cast<T>(microvision::common::sdk::unit::time::secondsPerHour));
    }
};
template<typename T>
struct Convert<angle::tick32, angle::degree, T>
{
    constexpr T operator()(const T& valInTick32s) const
    {
        return static_cast<T>(valInTick32s / static_cast<T>(angle::tick32sPerFullRotation / angle::degreesPerRotation));
    }
};
template<typename T>
struct Convert<angle::degree, angle::tick32, T>
{
    constexpr T operator()(const T& valInDegrees) const
    {
        return static_cast<T>(valInDegrees * static_cast<T>(angle::tick32sPerFullRotation / angle::degreesPerRotation));
    }
};

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
