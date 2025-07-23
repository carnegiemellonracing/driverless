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
//! \date May 8, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Vector2.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
// cm to meter
//==============================================================================

template<typename T>
static float convertFromCm(const T valueInCm)
{
    return float(valueInCm) * 1e-2F;
}

template<typename T>
static float convertFromCm(const T valueInCm, const T nanValue)
{
    return (valueInCm == nanValue) ? NaN : convertFromCm<T>(valueInCm);
}

template<typename T>
static Vector2<float> convertFromCm(const Vector2<T>& valueInCm)
{
    return Vector2<float>(convertFromCm<T>(valueInCm.getX()), convertFromCm<T>(valueInCm.getY()));
}

template<typename T>
static Vector2<float> convertFromCm(const Vector2<T>& valueInCm, const T nanValue)
{
    return Vector2<float>(convertFromCm<T>(valueInCm.getX(), nanValue), convertFromCm<T>(valueInCm.getY(), nanValue));
}

//==============================================================================

template<typename T>
static T convertToCm(const float valueInMeter)
{
    return T(roundf(valueInMeter * 100.0F));
}

template<typename T>
static T convertToCm(const float valueInMeter, const T nanValue)
{
    return std::isnan(valueInMeter) ? nanValue : convertToCm<T>(valueInMeter);
}

template<typename T>
static Vector2<T> convertToCm(const Vector2<float>& valueInMeter)
{
    return Vector2<T>(convertToCm<T>(valueInMeter.getX()), convertToCm<T>(valueInMeter.getY()));
}

template<typename T>
static Vector2<T> convertToCm(const Vector2<float>& valueInMeter, const T nanValue)
{
    return Vector2<T>(convertToCm<T>(valueInMeter.getX(), nanValue), convertToCm<T>(valueInMeter.getY(), nanValue));
}

//==============================================================================
// centidegrees to rad
//==============================================================================

template<typename T>
static float convertFromCentiDegrees(const T& valueInCentiDegrees)
{
    return float(valueInCentiDegrees) * 1e-2F * deg2radf;
}

template<typename T>
static float convertFromCentiDegrees(const T& valueInCentiDegrees, const T nanValue)
{
    return (valueInCentiDegrees == nanValue) ? NaN : convertFromCentiDegrees<T>(valueInCentiDegrees);
}

template<typename T>
static Vector2<float> convertFromCentiDegrees(const Vector2<T>& valueInCentiDegrees)
{
    return Vector2<float>(convertFromCentiDegrees<T>(valueInCentiDegrees.getX()),
                          convertFromCentiDegrees<T>(valueInCentiDegrees.getY()));
}

template<typename T>
static Vector2<float> convertFromCentiDegrees(const Vector2<T>& valueInCentiDegrees, const T nanValue)
{
    return Vector2<float>(convertFromCentiDegrees<T>(valueInCentiDegrees.getX(), nanValue),
                          convertFromCentiDegrees<T>(valueInCentiDegrees.getY(), nanValue));
}

//==============================================================================

template<typename T>
static T convertToCentiDegrees(const float valueInRad)
{
    return T(roundf(valueInRad / deg2radf * 100.0F));
}

template<typename T>
static T convertToCentiDegrees(const float valueInRad, const T nanValue)
{
    return std::isnan(valueInRad) ? nanValue : convertToCentiDegrees<T>(valueInRad);
}

template<typename T>
static Vector2<T> convertToCentiDegrees(const Vector2<float>& valueInRad)
{
    return Vector2<T>(convertToCentiDegrees<T>(valueInRad.getX()), convertToCentiDegrees<T>(valueInRad.getY()));
}

template<typename T>
static Vector2<T> convertToCentiDegrees(const Vector2<float>& valueInRad, const T nanValue)
{
    return Vector2<T>(convertToCentiDegrees<T>(valueInRad.getX(), nanValue),
                      convertToCentiDegrees<T>(valueInRad.getY(), nanValue));
}

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
