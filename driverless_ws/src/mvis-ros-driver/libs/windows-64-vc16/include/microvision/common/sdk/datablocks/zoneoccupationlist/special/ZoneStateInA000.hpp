//==============================================================================
//! \file
//!
//! \brief Enumeration defining possible states of a safety zone.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 29, 2025
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Enumeration of possible zone states.
//!
//! Defines the possible states a safety zone can be in during operation.
//------------------------------------------------------------------------------
enum class ZoneStateInA000 : uint8_t
{
    Occupied = 0, //!< Zone is occupied (default).
    Free     = 1, //!< Zone is free.
    Inactive = 2 //!< Zone is inactive.
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================