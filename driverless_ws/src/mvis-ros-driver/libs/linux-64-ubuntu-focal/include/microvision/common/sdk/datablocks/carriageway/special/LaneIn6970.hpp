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
//! \brief Lane which has a list of LaneSegments
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/LaneTemplate.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

// Forward declaration of template class.
class LaneSegmentIn6970;

//==============================================================================
//!\brief A LaneIn6970 represents a lane of a specific type within a \ref CarriageWaySegmentIn6970
//!
//! Each LaneIn6970 has a unique id within the parent \ref CarriageWaySegmentIn6970
//!
//! A LaneIn6970 holds a list of LaneSegmentIn6970 segments as well as pointers to preceding,
//! following and neighboring Lanes.
//!
//! The segmentation of a whole road is as following:
//!
//!\ref CarriageWayIn6970 \htmlonly&#8594;\endhtmlonly
//!\ref CarriageWaySegmentIn6970 \htmlonly&#8594;\endhtmlonly
//!\ref LaneIn6970 \htmlonly&#8594;\endhtmlonly
//!\ref LaneSegmentIn6970
//!
//! The connection is handled using smart pointers. Therefore it is not possible
//! to create an instance of this class, but calling \ref create will return
//! a shared pointer to a new LaneIn6970.
//!
//!\sa CarriageWayIn6970 \sa CarriageWaySegmentIn6970 \sa LaneSegmentIn6970
//!
//!\note The implementation of this class is done via templates (\sa LaneTemplate).
//------------------------------------------------------------------------------
using LaneIn6970 = LaneTemplate<LaneSegmentIn6970>;

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================

// Define classes previously declared as forward.
#include <microvision/common/sdk/datablocks/carriageway/special/LaneSegmentIn6970.hpp>
