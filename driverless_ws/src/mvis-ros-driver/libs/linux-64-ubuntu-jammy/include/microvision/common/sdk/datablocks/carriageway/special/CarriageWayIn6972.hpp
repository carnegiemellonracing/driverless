//==============================================================================
//! \file
//! \brief CarriageWay for datatype 0x6972 which stores CarriageWayIn6972Segments
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

#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayTemplate.hpp>
#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWaySegmentIn6972.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//!\brief A CarriageWayIn6972 represents one surface of a road and has a unique identifier.
//!
//! The identifier is a combination of the \ref CarriageWayType of  the road and a
//! number (e.g. A for \ref CarriageWayType motorway and 1 represents A1).
//!
//! In addition each CarriageWayIn6972 holds a list of segments. Within one segment,
//! the number of lanes is constant. If there are preceding and following segments,
//! these segments are linked against each other. It is therefore possible to
//! store multiple linked lists of segments within on CarriageWayIn6972 (e.g. for different
//! driving directions or if there are gaps between segments).
//!
//! A CarriageWay is always the highest representation of a road. The segmentation
//! is as following:
//!
//!\ref CarriageWayIn6972 \htmlonly&#8594;\endhtmlonly
//!\ref CarriageWaySegmentIn6972 \htmlonly&#8594;\endhtmlonly
//!\ref LaneIn6972 \htmlonly&#8594;\endhtmlonly
//!\ref LaneSegmentIn6972
//!
//! The connection is handled using smart pointers. Therefore it is not possible
//! to create an instance of this class, but calling \ref CarriageWayTemplate::create() "create()" will return
//! a shared pointer to a new CarriageWayIn6972.
//!
//!\sa CarriageWaySegmentIn6972 \sa LaneIn6972 \sa LaneSegmentIn6972
//!
//!\note The implementation of this class is done via templates (\sa CarriageWayTemplate).
//------------------------------------------------------------------------------
using CarriageWayIn6972 = CarriageWayTemplate<LaneSegmentIn6972>;

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
