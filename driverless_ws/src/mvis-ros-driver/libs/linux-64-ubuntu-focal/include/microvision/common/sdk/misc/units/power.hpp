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
namespace power {
//==============================================================================

// power constants
static constexpr uint32_t wattPerKilowatt              = unit::kilo;
static constexpr uint32_t horsepowerPerWattNumerator   = 7457; // British horsepower
static constexpr uint32_t horsepowerPerWattDenominator = 10;
static constexpr double psPerWatt                      = 735.49875; // german unit "Pferdestaerke"

// available and convertible power units
static constexpr uint64_t watt       = common::sdk::hash("watt");
static constexpr uint64_t kilowatt   = common::sdk::hash("kilowatt");
static constexpr uint64_t milliwatt  = common::sdk::hash("milliwatt");
static constexpr uint64_t horsepower = common::sdk::hash("horsepower");
static constexpr uint64_t ps         = common::sdk::hash("ps");

//==============================================================================
} // namespace power
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
struct Unit<power::watt>
{
    constexpr static const char* name() { return "watt"; }
    constexpr static const char* symbol() { return "W"; }
};
template<>
struct Unit<power::kilowatt>
{
    constexpr static const char* name() { return "kilowatt"; }
    constexpr static const char* symbol() { return "kW"; }
};
template<>
struct Unit<power::milliwatt>
{
    constexpr static const char* name() { return "milliwatt"; }
    constexpr static const char* symbol() { return "mW"; }
};
template<>
struct Unit<power::horsepower>
{
    constexpr static const char* name() { return "horsepower"; }
    constexpr static const char* symbol() { return "HP"; }
};
template<>
struct Unit<power::ps>
{
    constexpr static const char* name() { return "ps"; }
    constexpr static const char* symbol() { return "PS"; }
};

//==============================================================================

// conversions
template<typename T>
struct Convert<power::kilowatt, power::watt, T>
{
    constexpr T operator()(const T& valInKilowatt) const
    {
        return static_cast<T>(valInKilowatt * static_cast<T>(power::wattPerKilowatt));
    }
};
template<typename T>
struct Convert<power::watt, power::kilowatt, T>
{
    constexpr T operator()(const T& val) const { return static_cast<T>(val / static_cast<T>(power::wattPerKilowatt)); }
};
template<typename T>
struct Convert<power::milliwatt, power::watt, T>
{
    constexpr T operator()(const T& valInMilliwatt) const
    {
        return static_cast<T>(valInMilliwatt / static_cast<T>(power::wattPerKilowatt));
    }
};
template<typename T>
struct Convert<power::watt, power::milliwatt, T>
{
    constexpr T operator()(const T& val) const { return static_cast<T>(val * static_cast<T>(power::wattPerKilowatt)); }
};
template<typename T>
struct Convert<power::horsepower, power::watt, T>
{
    constexpr T operator()(const T& valInHorsepower) const
    {
        return static_cast<T>((valInHorsepower * static_cast<T>(power::horsepowerPerWattNumerator))
                              / static_cast<T>(power::horsepowerPerWattDenominator));
    }
};
template<typename T>
struct Convert<power::watt, power::horsepower, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>((val * static_cast<T>(power::horsepowerPerWattDenominator))
                              / static_cast<T>(power::horsepowerPerWattNumerator));
    }
};
template<typename T>
struct Convert<power::horsepower, power::kilowatt, T>
{
    constexpr T operator()(const T& valInHorsepower) const
    {
        return static_cast<T>((valInHorsepower * static_cast<T>(power::horsepowerPerWattNumerator))
                              / static_cast<T>(power::horsepowerPerWattDenominator)
                              * static_cast<T>(power::wattPerKilowatt));
    }
};
template<typename T>
struct Convert<power::kilowatt, power::horsepower, T>
{
    constexpr T operator()(const T& valInKilowatt) const
    {
        return static_cast<T>(
            (valInKilowatt * static_cast<T>(power::horsepowerPerWattDenominator * power::wattPerKilowatt))
            / static_cast<T>(power::horsepowerPerWattNumerator));
    }
};
template<typename T>
struct Convert<power::ps, power::watt, T>
{
    constexpr T operator()(const T& valInPferdestaerke) const
    {
        return static_cast<T>(valInPferdestaerke * static_cast<T>(power::psPerWatt));
    }
};
template<typename T>
struct Convert<power::watt, power::ps, T>
{
    constexpr T operator()(const T& val) const { return static_cast<T>(val / static_cast<T>(power::psPerWatt)); }
};
template<typename T>
struct Convert<power::ps, power::kilowatt, T>
{
    constexpr T operator()(const T& valInPferdestaerke) const
    {
        return static_cast<T>((valInPferdestaerke * static_cast<T>(power::psPerWatt))
                              / static_cast<T>(power::wattPerKilowatt));
    }
};
template<typename T>
struct Convert<power::kilowatt, power::ps, T>
{
    constexpr T operator()(const T& valInKilowatt) const
    {
        return static_cast<T>((valInKilowatt * static_cast<T>(power::wattPerKilowatt))
                              / static_cast<T>(power::psPerWatt));
    }
};

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
