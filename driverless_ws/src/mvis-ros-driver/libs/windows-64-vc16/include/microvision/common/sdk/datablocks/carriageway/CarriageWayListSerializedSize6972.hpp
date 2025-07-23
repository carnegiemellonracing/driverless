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
#include <microvision/common/sdk/datablocks/carriageway/special/CarriageWayList6972SerializedSize6972.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class CarriageWayListSerializedSize6972
{
public:
    static std::streamsize getSerializedSize(const CarriageWayList& c)
    {
        std::streamsize size = std::streamsize(sizeof(uint64_t)); // nb. of elements

        for (const lanes::CarriageWay::Ptr& cw : c.getCarriageWays())
        {
            size += getSerializedSize(*cw);
        }

        return size;
    }

    // Currently, the general data container is equal to 6972!
    static std::streamsize getSerializedSize(const lanes::CarriageWay& cw)
    {
        return CarriageWayList6972SerializedSize6972::getSerializedSize(cw);
    }

    static std::streamsize getSerializedSize(const lanes::CarriageWaySegment& segment)
    {
        return CarriageWayList6972SerializedSize6972::getSerializedSize(segment);
    }

    static std::streamsize getSerializedSize(const lanes::Lane& lane)
    {
        return CarriageWayList6972SerializedSize6972::getSerializedSize(lane);
    }

    static std::streamsize getSerializedSize(const lanes::LaneSegment& laneSegment)
    {
        return CarriageWayList6972SerializedSize6972::getSerializedSize(laneSegment);
    }

    static std::streamsize getSerializedSize(const lanes::LaneSupportPoint& laneSupportPoint)
    {
        return CarriageWayList6972SerializedSize6972::getSerializedSize(laneSupportPoint);
    }

}; //CarriageWayListSerializedSize6972

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
