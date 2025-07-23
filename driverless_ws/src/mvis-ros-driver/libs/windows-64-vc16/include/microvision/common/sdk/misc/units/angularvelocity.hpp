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
#include <microvision/common/sdk/misc/units/angle.hpp>
#include <microvision/common/sdk/misc/units/time.hpp>
#include <microvision/common/sdk/Math.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace unit {
namespace angularvelocity {
//==============================================================================

// angularvelocity constants

// available and convertible angularvelocity units
static constexpr uint64_t radianPerSecond          = common::sdk::hash("radianpersecond");
static constexpr uint64_t degreePerSecond          = common::sdk::hash("degreepersecond");
static constexpr uint64_t centidegreePerNanosecond = common::sdk::hash("centidegreepernanosecond");

//==============================================================================
} // namespace angularvelocity
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
struct Unit<angularvelocity::radianPerSecond>
{
    constexpr static const char* name() { return "radianPerSecond"; }
    constexpr static const char* symbol() { return "rad/s"; }
};
template<>
struct Unit<angularvelocity::degreePerSecond>
{
    constexpr static const char* name() { return "degreePerSecond"; }
    constexpr static const char* symbol() { return "°/s"; }
};
template<>
struct Unit<angularvelocity::centidegreePerNanosecond>
{
    constexpr static const char* name() { return "centidegreePerNanosecond"; }
    constexpr static const char* symbol() { return ""; }
};

//==============================================================================

// conversions
template<>
struct Convert<angularvelocity::radianPerSecond, angularvelocity::degreePerSecond, double>
{
    constexpr double operator()(const double& val) const { return val * static_cast<double>(rad2deg); }
};
template<>
struct Convert<angularvelocity::radianPerSecond, angularvelocity::degreePerSecond, float>
{
    constexpr float operator()(const float& val) const { return val * static_cast<float>(rad2degf); }
};
template<typename T>
struct Convert<angularvelocity::radianPerSecond, angularvelocity::degreePerSecond, T>
{
    static_assert(std::numeric_limits<T>::is_integer == false, "Integer type not useful for radian unit!");
    constexpr T operator()(const T& val) const
    {
        return (val * static_cast<T>(angle::degreesPerHalfRotation)) / static_cast<T>(pi);
    }
};

template<>
struct Convert<angularvelocity::degreePerSecond, angularvelocity::radianPerSecond, double>
{
    constexpr double operator()(const double& valInDegreePerSecond) const
    {
        return valInDegreePerSecond * static_cast<double>(deg2rad);
    }
};
template<>
struct Convert<angularvelocity::degreePerSecond, angularvelocity::radianPerSecond, float>
{
    constexpr float operator()(const float& valInDegreePerSecond) const
    {
        return valInDegreePerSecond * static_cast<float>(deg2radf);
    }
};
template<typename T>
struct Convert<angularvelocity::degreePerSecond, angularvelocity::radianPerSecond, T>
{
    static_assert(std::numeric_limits<T>::is_integer == false, "Integer type not useful for radian unit!");
    constexpr T operator()(const T& valInDegreePerSecond) const
    {
        return (valInDegreePerSecond * static_cast<T>(pi)) / static_cast<T>(angle::degreesPerHalfRotation);
    }
};

template<>
struct Convert<angularvelocity::radianPerSecond, angularvelocity::centidegreePerNanosecond, double>
{
    constexpr double operator()(const double& val) const
    {
        return Convert<angle::radian, angle::centidegree, double>()(val)
               / static_cast<double>(time::nanosecondsPerSecond);
    }
};
template<>
struct Convert<angularvelocity::radianPerSecond, angularvelocity::centidegreePerNanosecond, float>
{
    constexpr float operator()(const float& val) const
    {
        return Convert<angle::radian, angle::centidegree, float>()(val)
               / static_cast<float>(time::nanosecondsPerSecond);
    }
};
template<typename T>
struct Convert<angularvelocity::radianPerSecond, angularvelocity::centidegreePerNanosecond, T>
{
    static_assert(std::numeric_limits<T>::is_integer == false, "Integer type not useful for radian unit!");
    constexpr T operator()(const T& val) const
    {
        return Convert<angle::radian, angle::centidegree, T>()(val) / static_cast<T>(time::nanosecondsPerSecond);
    }
};

template<>
struct Convert<angularvelocity::centidegreePerNanosecond, angularvelocity::radianPerSecond, double>
{
    constexpr double operator()(const double& val) const
    {
        return Convert<angle::centidegree, angle::radian, double>()(val)
               * static_cast<double>(time::nanosecondsPerSecond);
    }
};
template<>
struct Convert<angularvelocity::centidegreePerNanosecond, angularvelocity::radianPerSecond, float>
{
    constexpr float operator()(const float& val) const
    {
        return Convert<angle::centidegree, angle::radian, float>()(val)
               * static_cast<float>(time::nanosecondsPerSecond);
    }
};
template<typename T>
struct Convert<angularvelocity::centidegreePerNanosecond, angularvelocity::radianPerSecond, T>
{
    static_assert(std::numeric_limits<T>::is_integer == false, "Integer type not useful for radian unit!");
    constexpr T operator()(const T& val) const
    {
        return Convert<angle::centidegree, angle::radian, T>()(val) * static_cast<T>(time::nanosecondsPerSecond);
    }
};

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
