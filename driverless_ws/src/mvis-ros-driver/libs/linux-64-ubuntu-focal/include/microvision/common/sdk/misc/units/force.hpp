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
namespace force {
//==============================================================================

// force constants
static constexpr uint32_t newtonsPerKilonewton          = unit::kilo;
static constexpr uint32_t poundforceToNewtonNumerator   = 100000;
static constexpr uint32_t poundforceToNewtonDenominator = mass::poundToKiloNumerator;

// available and convertible force units
static constexpr uint64_t newton     = common::sdk::hash("newton");
static constexpr uint64_t kilonewton = common::sdk::hash("kilonewton");
static constexpr uint64_t poundforce = common::sdk::hash("poundforce");

//==============================================================================
} // namespace force
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
struct Unit<force::newton>
{
    constexpr static const char* name() { return "newton"; }
    constexpr static const char* symbol() { return "N"; }
};
template<>
struct Unit<force::kilonewton>
{
    constexpr static const char* name() { return "kilonewton"; }
    constexpr static const char* symbol() { return "kN"; }
};
template<>
struct Unit<force::poundforce>
{
    constexpr static const char* name() { return "poundforce"; }
    constexpr static const char* symbol() { return "lbf"; }
};

//==============================================================================

// conversions
template<typename T>
struct Convert<force::kilonewton, force::newton, T>
{
    constexpr T operator()(const T& valInKiloNewton) const
    {
        return static_cast<T>(valInKiloNewton * static_cast<T>(force::newtonsPerKilonewton));
    }
};
template<typename T>
struct Convert<force::newton, force::kilonewton, T>
{
    constexpr T operator()(const T& val) const { return val / static_cast<T>(force::newtonsPerKilonewton); }
};

template<typename T>
struct Convert<force::poundforce, force::newton, T>
{
    constexpr T operator()(const T& valInPoundForce) const
    {
        return (valInPoundForce * static_cast<T>(force::poundforceToNewtonNumerator))
               / static_cast<T>(force::poundforceToNewtonDenominator);
    }
};
template<typename T>
struct Convert<force::newton, force::poundforce, T>
{
    constexpr T operator()(const T& val) const
    {
        return (val * static_cast<T>(force::poundforceToNewtonDenominator))
               / static_cast<T>(force::poundforceToNewtonNumerator);
    }
};

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
