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

#include <microvision/common/sdk/datablocks/carriageway/CarriageWayList.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CarriageWayListSerializedSize6970
{
public:
    static std::streamsize getSerializedSize(const CarriageWayList& c);
    static std::streamsize getSerializedSize(const lanes::CarriageWay& cw);
    static std::streamsize getSerializedSize(const lanes::CarriageWaySegment& segment);
    static std::streamsize getSerializedSize(const lanes::Lane& lane);
    static std::streamsize getSerializedSize(const lanes::LaneSegment& laneSegment);
    static std::streamsize getSerializedSize(const lanes::LaneSupportPoint& laneSupportPoint);
}; //CarriageWayListSerializedSize6970

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
