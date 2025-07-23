//==============================================================================
//! \file
//! \brief CarriageWay for datatype 0x6970 which stores CarriageWayIn6970Segments
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Oct 9, 2014
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayTemplate.hpp>
#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWaySegmentIn6970.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//!\brief A CarriageWayIn6970 represents one surface of a road and has a unique identifier.
//!
//! The identifier is a combination of the \ref microvision::common::sdk::lanes::CarriageWayType "CarriageWayType" of the road
//! and a number (e.g. A for CarriageWayType motorway and 1 represents A1).
//!
//! In addition each CarriageWayIn6970 holds a list of segments. Within one segment,
//! the number of lanes is constant. If there are preceding and following segments,
//! these segments are linked against each other. It is therefore possible to
//! store multiple linked lists of segments within on CarriageWayIn6970 (e.g. for different
//! driving directions or if there are gaps between segments).
//!
//! A CarriageWay is always the highest representation of a road. The segmentation
//! is as following:
//!
//!\ref microvision::common::sdk::lanes::CarriageWayIn6970 "CarriageWayIn6970" \htmlonly&#8594;\endhtmlonly
//!\ref microvision::common::sdk::lanes::CarriageWaySegmentIn6970 "CarriageWaySegmentIn6970" \htmlonly&#8594;\endhtmlonly
//!\ref microvision::common::sdk::lanes::LaneIn6970 "LaneIn6970" \htmlonly&#8594;\endhtmlonly
//!\ref microvision::common::sdk::lanes::LaneSegmentIn6970 "LaneSegmentIn6970"
//!
//! The connection is handled using smart pointers. Therefore it is not possible to create an instance of this class,
//! but calling \ref microvision::common::sdk::lanes::CarriageWayTemplate::create() "create()" will return a shared
//! pointer to a new CarriageWayIn6970.
//!
//!\sa CarriageWaySegmentIn6970 \sa LaneIn6970 \sa LaneSegmentIn6970
//!
//!\note The implementation of this class is done via templates
//! (\ref microvision::common::sdk::lanes::CarriageWayTemplate "CarriageWayTemplate").
//------------------------------------------------------------------------------
using CarriageWayIn6970 = CarriageWayTemplate<LaneSegmentIn6970>;

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
