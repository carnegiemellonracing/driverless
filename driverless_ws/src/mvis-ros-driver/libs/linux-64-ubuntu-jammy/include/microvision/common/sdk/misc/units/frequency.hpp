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
namespace frequency {
//==============================================================================

//==============================================================================
// frequency constants
// available and convertible frequency units
//------------------------------------------------------------------------------
static constexpr uint64_t hertz = common::sdk::hash("hertz");

//==============================================================================
} // namespace frequency
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

//==============================================================================
// names and symbols
//------------------------------------------------------------------------------
template<>
struct Unit<frequency::hertz>
{
    constexpr static const char* name() { return "hertz"; }
    constexpr static const char* symbol() { return "Hz"; }
};

//==============================================================================
// conversions
//------------------------------------------------------------------------------

//==============================================================================
} // namespace unit
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
