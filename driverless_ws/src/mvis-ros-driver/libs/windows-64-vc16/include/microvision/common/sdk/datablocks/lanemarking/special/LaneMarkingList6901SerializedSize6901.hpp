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

#include <microvision/common/sdk/datablocks/lanemarking/special/LaneMarkingList6901.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Here is the size defined for serialized laneMarkingsLists.
//------------------------------------------------------------------------------
class LaneMarkingList6901SerializedSize6901
{
public:
    static std::streamsize getSerializedSize(const LaneMarkingList6901& markingList);
    static std::streamsize getSerializedSize(const lanes::LaneMarkingIn6901& laneMarking);
    static std::streamsize getSerializedSize(const lanes::LaneMarkingSegmentIn6901& segments);

}; //LaneMarkingList6901SerializedSize6901

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
