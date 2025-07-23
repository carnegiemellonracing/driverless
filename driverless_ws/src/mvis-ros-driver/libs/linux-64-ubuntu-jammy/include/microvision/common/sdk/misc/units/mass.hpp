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

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace unit {
namespace mass {
//==============================================================================

// mass constants
static constexpr uint32_t kilosPerTon            = unit::kilo;
static constexpr uint32_t gramsPerKilogram       = unit::kilo;
static constexpr uint32_t millisPerGram          = unit::milli;
static constexpr uint32_t ouncesPerPound         = 16;
static constexpr uint32_t poundToKiloNumerator   = 45359237;
static constexpr uint32_t poundToKiloDenominator = 100000000;

// available and convertible mass units
static constexpr uint64_t kilogram  = common::sdk::hash("kilogram");
static constexpr uint64_t ton       = common::sdk::hash("ton");
static constexpr uint64_t gram      = common::sdk::hash("gram");
static constexpr uint64_t milligram = common::sdk::hash("milligram");
static constexpr uint64_t pound     = common::sdk::hash("pound");
static constexpr uint64_t ounce     = common::sdk::hash("ounce");

//==============================================================================
} // namespace mass
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
struct Unit<mass::kilogram>
{
    constexpr static const char* name() { return "kilogram"; }
    constexpr static const char* symbol() { return "kg"; }
};
template<>
struct Unit<mass::ton>
{
    constexpr static const char* name() { return "ton"; }
    constexpr static const char* symbol() { return "t"; }
};
template<>
struct Unit<mass::gram>
{
    constexpr static const char* name() { return "gram"; }
    constexpr static const char* symbol() { return "g"; }
};
template<>
struct Unit<mass::milligram>
{
    constexpr static const char* name() { return "milligram"; }
    constexpr static const char* symbol() { return "mg"; }
};

template<>
struct Unit<mass::pound>
{
    constexpr static const char* name() { return "pound"; }
    constexpr static const char* symbol() { return "lb"; }
};

template<>
struct Unit<mass::ounce>
{
    constexpr static const char* name() { return "ounce"; }
    constexpr static const char* symbol() { return "oz"; }
};

//==============================================================================

// conversions
template<typename T>
struct Convert<mass::ton, mass::kilogram, T>
{
    constexpr T operator()(const T& valInTons) const
    {
        return static_cast<T>(valInTons * static_cast<T>(mass::kilosPerTon));
    }
};
template<typename T>
struct Convert<mass::kilogram, mass::ton, T>
{
    constexpr T operator()(const T& val) const { return static_cast<T>(val / static_cast<T>(mass::kilosPerTon)); }
};
template<typename T>
struct Convert<mass::gram, mass::kilogram, T>
{
    constexpr T operator()(const T& valInGrams) const
    {
        return static_cast<T>(valInGrams / static_cast<T>(mass::gramsPerKilogram));
    }
};
template<typename T>
struct Convert<mass::kilogram, mass::gram, T>
{
    constexpr T operator()(const T& val) const { return static_cast<T>(val * static_cast<T>(mass::gramsPerKilogram)); }
};
template<typename T>
struct Convert<mass::milligram, mass::gram, T>
{
    constexpr T operator()(const T& valInMilligrams) const
    {
        return static_cast<T>(valInMilligrams / static_cast<T>(mass::millisPerGram));
    }
};
template<typename T>
struct Convert<mass::gram, mass::milligram, T>
{
    constexpr T operator()(const T& valInGrams) const
    {
        return static_cast<T>(valInGrams * static_cast<T>(mass::millisPerGram));
    }
};
template<typename T>
struct Convert<mass::milligram, mass::kilogram, T>
{
    constexpr T operator()(const T& valInMilligrams) const
    {
        return static_cast<T>(valInMilligrams / static_cast<T>(mass::millisPerGram * mass::millisPerGram));
    }
};
template<typename T>
struct Convert<mass::kilogram, mass::milligram, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val * static_cast<T>(mass::millisPerGram * mass::millisPerGram));
    }
};
template<typename T>
struct Convert<mass::pound, mass::kilogram, T>
{
    constexpr T operator()(const T& valInPounds) const
    {
        return static_cast<T>((valInPounds * static_cast<T>(mass::poundToKiloNumerator))
                              / static_cast<T>(mass::poundToKiloDenominator));
    }
};
template<typename T>
struct Convert<mass::kilogram, mass::pound, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>((val * static_cast<T>(mass::poundToKiloDenominator))
                              / static_cast<T>(mass::poundToKiloNumerator));
    }
};
template<typename T>
struct Convert<mass::ounce, mass::pound, T>
{
    constexpr T operator()(const T& valInOunces) const
    {
        return static_cast<T>(valInOunces / static_cast<T>(mass::ouncesPerPound));
    }
};
template<typename T>
struct Convert<mass::pound, mass::ounce, T>
{
    constexpr T operator()(const T& valInPounds) const
    {
        return static_cast<T>(valInPounds * static_cast<T>(mass::ouncesPerPound));
    }
};

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
