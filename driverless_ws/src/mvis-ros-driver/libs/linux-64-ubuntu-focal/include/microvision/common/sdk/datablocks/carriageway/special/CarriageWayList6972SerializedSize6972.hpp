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

#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayList6972.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CarriageWayList6972SerializedSize6972
{
public:
    static std::streamsize getSerializedSize(const CarriageWayList6972& c);
    static std::streamsize getSerializedSize(const lanes::CarriageWayIn6972& cw);
    static std::streamsize getSerializedSize(const lanes::CarriageWaySegmentIn6972& segment);
    static std::streamsize getSerializedSize(const lanes::LaneIn6972& lane);
    static std::streamsize getSerializedSize(const lanes::LaneSegmentIn6972& laneSegment);
    static std::streamsize getSerializedSize(const lanes::LaneSupportPointIn6972& laneSupportPoint);
}; //CarriageWayList6972SerializedSize6972

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
