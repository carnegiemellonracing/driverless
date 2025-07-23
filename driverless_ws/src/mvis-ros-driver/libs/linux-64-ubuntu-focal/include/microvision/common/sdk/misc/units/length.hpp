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
namespace length {
//==============================================================================

// length constants
static constexpr uint32_t metersPerKilometer       = unit::kilo;
static constexpr uint32_t decimetersPerMeter       = 10;
static constexpr uint32_t centimetersPerMeter      = unit::centi;
static constexpr uint32_t millimetersPerCentimeter = 10;
static constexpr uint32_t micrometersPerMeter      = unit::micro;
static constexpr double nanometersPerMeter         = unit::nano;
static constexpr uint32_t feetPerMeterNumerator    = 381;
static constexpr uint32_t feetPerMeterDenominator  = 1250;
static constexpr uint32_t feetPerMile              = 5280;
static constexpr uint32_t feetPerYard              = 3;
static constexpr uint32_t inchesPerFoot            = 12;

// available and convertible length units
static constexpr uint64_t meter      = common::sdk::hash("meter");
static constexpr uint64_t kilometer  = common::sdk::hash("kilometer");
static constexpr uint64_t decimeter  = common::sdk::hash("decimeter");
static constexpr uint64_t centimeter = common::sdk::hash("centimeter");
static constexpr uint64_t millimeter = common::sdk::hash("millimeter");
static constexpr uint64_t micrometer = common::sdk::hash("micrometer");
static constexpr uint64_t nanometer  = common::sdk::hash("nanometer");
static constexpr uint64_t foot       = common::sdk::hash("foot");
static constexpr uint64_t mile       = common::sdk::hash("mile");
static constexpr uint64_t inch       = common::sdk::hash("inch");
static constexpr uint64_t yard       = common::sdk::hash("yard");

//==============================================================================
} // namespace length
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
struct Unit<length::meter>
{
    constexpr static const char* name() { return "meter"; }
    constexpr static const char* symbol() { return "m"; }
};
template<>
struct Unit<length::kilometer>
{
    constexpr static const char* name() { return "kilometer"; }
    constexpr static const char* symbol() { return "km"; }
};
template<>
struct Unit<length::decimeter>
{
    constexpr static const char* name() { return "decimeter"; }
    constexpr static const char* symbol() { return "dm"; }
};
template<>
struct Unit<length::centimeter>
{
    constexpr static const char* name() { return "centimeter"; }
    constexpr static const char* symbol() { return "cm"; }
};
template<>
struct Unit<length::millimeter>
{
    constexpr static const char* name() { return "millimeter"; }
    constexpr static const char* symbol() { return "mm"; }
};
template<>
struct Unit<length::micrometer>
{
    constexpr static const char* name() { return "micrometer"; }
    constexpr static const char* symbol() { return "\230m"; }
};
template<>
struct Unit<length::nanometer>
{
    constexpr static const char* name() { return "nanometer"; }
    constexpr static const char* symbol() { return "nm"; }
};

template<>
struct Unit<length::foot>
{
    constexpr static const char* name() { return "foot"; }
    constexpr static const char* symbol() { return "ft"; }
};
template<>
struct Unit<length::mile>
{
    constexpr static const char* name() { return "mile"; }
    constexpr static const char* symbol() { return "mi"; }
};
template<>
struct Unit<length::inch>
{
    constexpr static const char* name() { return "inch"; }
    constexpr static const char* symbol() { return "in"; }
};
template<>
struct Unit<length::yard>
{
    constexpr static const char* name() { return "yard"; }
    constexpr static const char* symbol() { return "yd"; }
};

//==============================================================================

// conversions
template<typename T>
struct Convert<length::kilometer, length::meter, T>
{
    constexpr T operator()(const T& valInKilometers) const
    {
        return static_cast<T>(valInKilometers * static_cast<T>(length::metersPerKilometer));
    }
};
template<typename T>
struct Convert<length::meter, length::kilometer, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val / static_cast<T>(length::metersPerKilometer));
    }
};
template<typename T>
struct Convert<length::centimeter, length::meter, T>
{
    constexpr T operator()(const T& valInCentimeters) const
    {
        return static_cast<T>(valInCentimeters / static_cast<T>(length::centimetersPerMeter));
    }
};
template<typename T>
struct Convert<length::decimeter, length::meter, T>
{
    constexpr T operator()(const T& valInDecimeters) const
    {
        return valInDecimeters / static_cast<T>(length::decimetersPerMeter);
    }
};
template<typename T>
struct Convert<length::meter, length::centimeter, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val * static_cast<T>(length::centimetersPerMeter));
    }
};
template<typename T>
struct Convert<length::millimeter, length::meter, T>
{
    constexpr T operator()(const T& valInMillimeters) const
    {
        return static_cast<T>(valInMillimeters //
                              / static_cast<T>(length::millimetersPerCentimeter)
                              / static_cast<T>(length::centimetersPerMeter));
    }
};
template<typename T>
struct Convert<length::meter, length::millimeter, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val //
                              * static_cast<T>(length::centimetersPerMeter) //
                              * static_cast<T>(length::millimetersPerCentimeter));
    }
};
template<typename T>
struct Convert<length::centimeter, length::millimeter, T>
{
    constexpr T operator()(const T& valInCentimeters) const
    {
        return static_cast<T>(valInCentimeters * static_cast<T>(length::millimetersPerCentimeter));
    }
};
template<typename T>
struct Convert<length::millimeter, length::centimeter, T>
{
    constexpr T operator()(const T& valInMillimeter) const
    {
        return static_cast<T>(valInMillimeter / static_cast<T>(length::millimetersPerCentimeter));
    }
};
template<typename T>
struct Convert<length::micrometer, length::meter, T>
{
    constexpr T operator()(const T& valInMicrometers) const
    {
        return static_cast<T>(valInMicrometers / static_cast<T>(length::micrometersPerMeter));
    }
};
template<typename T>
struct Convert<length::meter, length::micrometer, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val * static_cast<T>(length::micrometersPerMeter));
    }
};
template<typename T>
struct Convert<length::nanometer, length::meter, T>
{
    constexpr T operator()(const T& valInNanometers) const
    {
        return static_cast<T>(valInNanometers / static_cast<T>(length::nanometersPerMeter));
    }
};
template<typename T>
struct Convert<length::meter, length::nanometer, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val * static_cast<T>(length::nanometersPerMeter));
    }
};
template<typename T>
struct Convert<length::foot, length::meter, T>
{
    constexpr T operator()(const T& valInFeet) const
    {
        return static_cast<T>((valInFeet * static_cast<T>(length::feetPerMeterNumerator))
                              / static_cast<T>(length::feetPerMeterDenominator));
    }
};
template<typename T>
struct Convert<length::meter, length::foot, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>((val * static_cast<T>(length::feetPerMeterDenominator))
                              / static_cast<T>(length::feetPerMeterNumerator));
    }
};
template<typename T>
struct Convert<length::mile, length::foot, T>
{
    constexpr T operator()(const T& valInMiles) const
    {
        return static_cast<T>(valInMiles * static_cast<T>(length::feetPerMile));
    }
};
template<typename T>
struct Convert<length::foot, length::mile, T>
{
    constexpr T operator()(const T& valInFeet) const
    {
        return static_cast<T>(valInFeet / static_cast<T>(length::feetPerMile));
    }
};
template<typename T>
struct Convert<length::yard, length::foot, T>
{
    constexpr T operator()(const T& valInYards) const
    {
        return static_cast<T>(valInYards * static_cast<T>(length::feetPerYard));
    }
};
template<typename T>
struct Convert<length::foot, length::yard, T>
{
    constexpr T operator()(const T& valInFeet) const
    {
        return static_cast<T>(valInFeet / static_cast<T>(length::feetPerYard));
    }
};
template<typename T>
struct Convert<length::inch, length::foot, T>
{
    constexpr T operator()(const T& valInInches) const
    {
        return static_cast<T>(valInInches / static_cast<T>(length::inchesPerFoot));
    }
};
template<typename T>
struct Convert<length::foot, length::inch, T>
{
    constexpr T operator()(const T& valInFeet) const
    {
        return static_cast<T>(valInFeet * static_cast<T>(length::inchesPerFoot));
    }
};

template<typename T>
struct Convert<length::mile, length::kilometer, T>
{
    constexpr T operator()(const T& valInMiles) const
    {
        return Convert<length::meter, length::kilometer, T>()(
            Convert<length::foot, length::meter, T>()(Convert<length::mile, length::foot, T>()(valInMiles)));
    }
};
template<typename T>
struct Convert<length::kilometer, length::mile, T>
{
    constexpr T operator()(const T& valInKilometers) const
    {
        return Convert<length::foot, length::mile, T>()(
            Convert<length::meter, length::foot, T>()(Convert<length::kilometer, length::meter, T>()(valInKilometers)));
    }
};

template<typename T>
struct Convert<length::mile, length::meter, T>
{
    constexpr T operator()(const T& valInMiles) const
    {
        // for this velocity conversion the hours are not relevant and do not change
        return Convert<length::foot, length::meter, T>()(Convert<length::mile, length::foot, T>()(valInMiles));
    }
};
template<typename T>
struct Convert<length::meter, length::mile, T>
{
    constexpr T operator()(const T& valInMeters) const
    {
        // for this velocity conversion the hours are not relevant and do not change
        return Convert<length::foot, length::mile, T>()(Convert<length::meter, length::foot, T>()(valInMeters));
    }
};

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
// ==============================================================================
