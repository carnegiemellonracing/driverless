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

#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayIn6972.hpp>
#include <microvision/common/sdk/datablocks/carriageway/CarriageWaySegment.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//!\brief A CarriageWay represents one surface of a road and has a unique identifier.
//!
//! The identifier is a combination of the \ref CarriageWayType of  the road and a
//! number (e.g. A for \ref CarriageWayType motorway and 1 represents A1).
//!
//! In addition each CarriageWay holds a list of segments. Within one segment,
//! the number of lanes is constant. If there are preceding and following segments,
//! these segments are linked against each other. It is therefore possible to
//! store multiple linked lists of segments within on CarriageWay (e.g. for different
//! driving directions or if there are gaps between segments).
//!
//! A CarriageWay is always the highest representation of a road. The segmentation
//! is as following:
//!
//!\ref CarriageWay \htmlonly&#8594;\endhtmlonly
//!\ref CarriageWaySegment \htmlonly&#8594;\endhtmlonly
//!\ref Lane \htmlonly&#8594;\endhtmlonly
//!\ref LaneSegment
//!
//! The connection is handled using smart pointers. Therefore it is not possible
//! to create an instance of this class, but calling \ref CarriageWayIn6972::create() "create()" will return
//! a shared pointer to a new CarriageWay.
//!
//!\sa CarriageWaySegment \sa Lane \sa LaneSegment
//!
//!\note The implementation of this class is done via templates (\sa CarriageWayTemplate).
//------------------------------------------------------------------------------
using CarriageWay = CarriageWayIn6972;

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
