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
//! \date Dec 13, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayList6970.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CarriageWayList6970SerializedSize6970
{
public:
    static std::streamsize getSerializedSize(const CarriageWayList6970& c);
    static std::streamsize getSerializedSize(const lanes::CarriageWayIn6970& cw);
    static std::streamsize getSerializedSize(const lanes::CarriageWaySegmentIn6970& segment);
    static std::streamsize getSerializedSize(const lanes::LaneIn6970& lane);
    static std::streamsize getSerializedSize(const lanes::LaneSegmentIn6970& laneSegment);
    static std::streamsize getSerializedSize(const lanes::LaneSupportPointIn6970& laneSupportPoint);
}; //CarriageWayList6970SerializedSize6970

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
