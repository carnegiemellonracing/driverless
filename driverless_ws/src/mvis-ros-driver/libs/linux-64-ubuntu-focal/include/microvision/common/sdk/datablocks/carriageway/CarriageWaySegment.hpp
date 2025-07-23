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
//! \date Dec 19, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWaySegmentIn6972.hpp>
#include <microvision/common/sdk/datablocks/carriageway/Lane.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//!\brief A CarriageWaySegment represents a single segment of a \ref CarriageWay.
//!
//! Each CarriageWaySegment has a unique id within the parent \ref CarriageWay.
//!
//! A \ref CarriageWay holds a constant number of lanes of type Lane. The segmentation of a whole
//! \ref CarriageWay is as following:
//!
//!\ref CarriageWay \htmlonly&#8594;\endhtmlonly
//!\ref CarriageWaySegment \htmlonly&#8594;\endhtmlonly
//!\ref Lane \htmlonly&#8594;\endhtmlonly
//!\ref LaneSegment
//!
//! The connection is handled using smart pointers. Therefore it is not possible
//! to create an instance of this class, but calling \ref create() will return
//! a shared pointer to a new CarriageWaySegment.
//!
//!\sa CarriageWay \sa Lane \sa LaneSegment
//!
//!\note The implementation of this class is done via templates (\sa CarriageWaySegmentTemplate).
//------------------------------------------------------------------------------
using CarriageWaySegment = CarriageWaySegmentIn6972;

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
