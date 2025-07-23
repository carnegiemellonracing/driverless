//==============================================================================
//! \file
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Oct 9, 2014
//! \brief Support point for a LaneSegment
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/LaneSupportPointIn6970.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//!\brief This class represents a support point of a \ref LaneSegmentIn6972.
//!
//! A point holds information about gps position and heading and width
//! (more precisely the offsets to the left and right bounding line).
//------------------------------------------------------------------------------
using LaneSupportPointIn6972 = LaneSupportPointIn6970;

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
