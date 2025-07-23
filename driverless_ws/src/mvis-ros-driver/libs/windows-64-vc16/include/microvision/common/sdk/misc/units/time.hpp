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
namespace time {
//==============================================================================

// time constants
static constexpr uint32_t nanosecondsPerSecond       = unit::nano;
static constexpr uint32_t microsecondsPerSecond      = unit::micro;
static constexpr uint32_t millisecondsPerSecond      = unit::milli;
static constexpr uint32_t nanosecondsPerMicroseconds = unit::milli;
static constexpr uint32_t secondsPerMinute           = 60;
static constexpr uint32_t minutesPerHour             = 60;
static constexpr uint32_t secondsPerHour             = secondsPerMinute * minutesPerHour;
static constexpr uint32_t hoursPerDay                = 24;
static constexpr uint32_t daysPerYear                = 365;

// available and convertible time units
static constexpr uint64_t second      = common::sdk::hash("second");
static constexpr uint64_t day         = common::sdk::hash("day");
static constexpr uint64_t hour        = common::sdk::hash("hour");
static constexpr uint64_t minute      = common::sdk::hash("minute");
static constexpr uint64_t millisecond = common::sdk::hash("millisecond");
static constexpr uint64_t microsecond = common::sdk::hash("microsecond");
static constexpr uint64_t nanosecond  = common::sdk::hash("nanosecond");
static constexpr uint64_t year        = common::sdk::hash("year");

//==============================================================================
} // namespace time
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
struct Unit<time::second>
{
    constexpr static const char* name() { return "second"; }
    constexpr static const char* symbol() { return "s"; }
};
template<>
struct Unit<time::day>
{
    constexpr static const char* name() { return "day"; }
    constexpr static const char* symbol() { return "d"; }
};
template<>
struct Unit<time::hour>
{
    constexpr static const char* name() { return "hour"; }
    constexpr static const char* symbol() { return "h"; }
};
template<>
struct Unit<time::minute>
{
    constexpr static const char* name() { return "minute"; }
    constexpr static const char* symbol() { return "m"; }
};
template<>
struct Unit<time::millisecond>
{
    constexpr static const char* name() { return "millisecond"; }
    constexpr static const char* symbol() { return "ms"; }
};
template<>
struct Unit<time::microsecond>
{
    constexpr static const char* name() { return "microsecond"; }
    constexpr static const char* symbol() { return "us"; }
};
template<>
struct Unit<time::nanosecond>
{
    constexpr static const char* name() { return "nanosecond"; }
    constexpr static const char* symbol() { return "ns"; }
};
template<>
struct Unit<time::year>
{
    constexpr static const char* name() { return "year"; }
    constexpr static const char* symbol() { return "y"; }
};

//==============================================================================

// conversions
template<typename T>
struct Convert<time::day, time::second, T>
{
    constexpr T operator()(const T& valInDays) const
    {
        return static_cast<T>(valInDays
                              * static_cast<T>((time::secondsPerMinute * time::minutesPerHour) * time::hoursPerDay));
    }
};
template<typename T>
struct Convert<time::second, time::day, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val
                              / static_cast<T>((time::secondsPerMinute * time::minutesPerHour) * time::hoursPerDay));
    }
};
template<typename T>
struct Convert<time::hour, time::second, T>
{
    constexpr T operator()(const T& valInHours) const
    {
        return static_cast<T>(valInHours * static_cast<T>(time::secondsPerMinute * time::minutesPerHour));
    }
};
template<typename T>
struct Convert<time::second, time::hour, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val / static_cast<T>(time::secondsPerMinute * time::minutesPerHour));
    }
};
template<typename T>
struct Convert<time::hour, time::day, T>
{
    constexpr T operator()(const T& valInHours) const
    {
        return static_cast<T>(valInHours / static_cast<T>(time::hoursPerDay));
    }
};
template<typename T>
struct Convert<time::day, time::hour, T>
{
    constexpr T operator()(const T& valInDays) const
    {
        return static_cast<T>(valInDays * static_cast<T>(time::hoursPerDay));
    }
};
template<typename T>
struct Convert<time::minute, time::second, T>
{
    constexpr T operator()(const T& valInMinutes) const
    {
        return static_cast<T>(valInMinutes * static_cast<T>(time::secondsPerMinute));
    }
};
template<typename T>
struct Convert<time::second, time::minute, T>
{
    constexpr T operator()(const T& val) const { return static_cast<T>(val / static_cast<T>(time::secondsPerMinute)); }
};
template<typename T>
struct Convert<time::hour, time::minute, T>
{
    constexpr T operator()(const T& valInHours) const
    {
        return static_cast<T>(valInHours * static_cast<T>(time::minutesPerHour));
    }
};
template<typename T>
struct Convert<time::minute, time::hour, T>
{
    constexpr T operator()(const T& valInMinutes) const
    {
        return static_cast<T>(valInMinutes / static_cast<T>(time::minutesPerHour));
    }
};
template<typename T>
struct Convert<time::millisecond, time::second, T>
{
    constexpr T operator()(const T& valInMilliseconds) const
    {
        return static_cast<T>(valInMilliseconds / static_cast<T>(time::millisecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::second, time::millisecond, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val * static_cast<T>(time::millisecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::microsecond, time::second, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val / static_cast<T>(time::microsecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::second, time::microsecond, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val * static_cast<T>(time::microsecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::microsecond, time::millisecond, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val / static_cast<T>(time::microsecondsPerSecond / time::millisecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::millisecond, time::microsecond, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val * static_cast<T>(time::microsecondsPerSecond / time::millisecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::nanosecond, time::second, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val / static_cast<T>(time::nanosecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::second, time::nanosecond, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val * static_cast<T>(time::nanosecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::nanosecond, time::millisecond, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val / static_cast<T>(time::nanosecondsPerSecond / time::millisecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::millisecond, time::nanosecond, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val * static_cast<T>(time::nanosecondsPerSecond / time::millisecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::nanosecond, time::microsecond, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val / static_cast<T>(time::nanosecondsPerSecond / time::microsecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::microsecond, time::nanosecond, T>
{
    constexpr T operator()(const T& val) const
    {
        return static_cast<T>(val * static_cast<T>(time::nanosecondsPerSecond / time::microsecondsPerSecond));
    }
};
template<typename T>
struct Convert<time::year, time::day, T>
{
    constexpr T operator()(const T& valInYears) const
    {
        return static_cast<T>(valInYears * static_cast<T>(time::daysPerYear));
    }
};
template<typename T>
struct Convert<time::day, time::year, T>
{
    constexpr T operator()(const T& valInDays) const
    {
        return static_cast<T>(valInDays / static_cast<T>(time::daysPerYear));
    }
};

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
