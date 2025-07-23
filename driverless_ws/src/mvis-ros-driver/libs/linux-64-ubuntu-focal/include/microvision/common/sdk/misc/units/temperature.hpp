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

#include <limits>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace unit {
namespace temperature {
//==============================================================================

// temperature constants
static constexpr double absoluteZero                     = -273.15;
static constexpr uint32_t fahrenheitZero                 = 32;
static constexpr uint32_t celsiusToFahrenheitNumerator   = 9;
static constexpr uint32_t celsiusToFahrenheitDenominator = 5;

// available and convertible temperature units
static constexpr uint64_t kelvin     = common::sdk::hash("kelvin");
static constexpr uint64_t celsius    = common::sdk::hash("celsius");
static constexpr uint64_t fahrenheit = common::sdk::hash("fahrenheit");

//==============================================================================
} // namespace temperature
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
struct Unit<temperature::kelvin>
{
    constexpr static const char* name() { return "kelvin"; }
    constexpr static const char* symbol() { return "K"; }
};
template<>
struct Unit<temperature::celsius>
{
    constexpr static const char* name() { return "celsius"; }
    constexpr static const char* symbol() { return "°C"; }
};
template<>
struct Unit<temperature::fahrenheit>
{
    constexpr static const char* name() { return "fahrenheit"; }
    constexpr static const char* symbol() { return "°F"; }
};

//==============================================================================

// conversions
template<typename T>
struct Convert<temperature::kelvin, temperature::celsius, T>
{
    static_assert((std::numeric_limits<T>::lowest() <= temperature::absoluteZero) && std::numeric_limits<T>::is_signed,
                  "Type not usable for conversion!");
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val + static_cast<T>(temperature::absoluteZero));
    }
};
template<typename T>
struct Convert<temperature::celsius, temperature::kelvin, T>
{
    static_assert((std::numeric_limits<T>::lowest() <= temperature::absoluteZero) && std::numeric_limits<T>::is_signed,
                  "Type not usable for conversion!");
    constexpr T operator()(const T& valInCelsius) const
    {
        return static_cast<T>(valInCelsius - static_cast<T>(temperature::absoluteZero));
    }
};
template<typename T>
struct Convert<temperature::celsius, temperature::fahrenheit, T>
{
    constexpr T operator()(const T& valInCelsius) const
    {
        return static_cast<T>(((valInCelsius * static_cast<T>(temperature::celsiusToFahrenheitNumerator))
                               / static_cast<T>(temperature::celsiusToFahrenheitDenominator))
                              + static_cast<T>(temperature::fahrenheitZero));
    }
};
template<typename T>
struct Convert<temperature::fahrenheit, temperature::celsius, T>
{
    constexpr T operator()(const T& valInFahrenheit) const
    {
        return static_cast<T>(((valInFahrenheit - static_cast<T>(temperature::fahrenheitZero))
                               * static_cast<T>(temperature::celsiusToFahrenheitDenominator))
                              / static_cast<T>(temperature::celsiusToFahrenheitNumerator));
    }
};
template<typename T>
struct Convert<temperature::kelvin, temperature::fahrenheit, T>
{
    constexpr T operator()(const T& val) const
    {
        return Convert<temperature::celsius, temperature::fahrenheit, T>()(
            Convert<temperature::kelvin, temperature::celsius, T>()(val));
    }
};
template<typename T>
struct Convert<temperature::fahrenheit, temperature::kelvin, T>
{
    constexpr T operator()(const T& valInFahrenheit) const
    {
        return Convert<temperature::celsius, temperature::kelvin, T>()(
            Convert<temperature::fahrenheit, temperature::celsius, T>()(valInFahrenheit));
    }
};

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
