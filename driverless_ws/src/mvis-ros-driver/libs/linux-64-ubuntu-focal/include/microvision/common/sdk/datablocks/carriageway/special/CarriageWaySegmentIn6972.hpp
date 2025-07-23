//==============================================================================
//! \file
//! \brief CarriageWaySegment6972 which has a constant number of lanes of type LaneIn6972
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Aug 16, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWaySegmentTemplate.hpp>
#include <microvision/common/sdk/datablocks/carriageway/special/LaneIn6972.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//!\brief A CarriageWaySegment6972 represents a single segment of a \ref CarriageWayIn6972.
//!
//! Each CarriageWaySegment6972 has a unique id within the parent \ref CarriageWayIn6972.
//!
//! A \ref CarriageWayIn6972 holds a constant number of lanes of type LaneIn6972. The segmentation of a whole
//! \ref CarriageWayIn6972 is as following:
//!
//!\ref CarriageWayIn6972 \htmlonly&#8594;\endhtmlonly
//!\ref CarriageWaySegmentIn6972 \htmlonly&#8594;\endhtmlonly
//!\ref LaneIn6972 \htmlonly&#8594;\endhtmlonly
//!\ref LaneSegmentIn6972
//!
//! The connection is handled using smart pointers. Therefore it is not possible
//! to create an instance of this class, but calling \ref create() will return
//! a shared pointer to a new CarriageWaySegmentIn6972.
//!
//!\sa CarriageWayIn6972 \sa LaneIn6972 \sa LaneSegmentIn6972
//!
//!\note The implementation of this class is done via templates (\sa CarriageWaySegmentTemplate).
//------------------------------------------------------------------------------
using CarriageWaySegmentIn6972 = CarriageWaySegmentTemplate<LaneSegmentIn6972>;

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
