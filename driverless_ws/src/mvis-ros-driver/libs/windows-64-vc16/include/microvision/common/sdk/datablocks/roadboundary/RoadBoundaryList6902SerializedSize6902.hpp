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
//! \date Okt 19, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/roadboundary/RoadBoundaryList6902.hpp>
#include <microvision/common/sdk/datablocks/roadboundary/RoadBoundaryIn6902.hpp>
#include <microvision/common/sdk/datablocks/roadboundary/RoadBoundarySegmentIn6902.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Here is the size defined for serialized RoadBoundaryLists.
//------------------------------------------------------------------------------

class RoadBoundaryList6902SerializedSize6902
{
public:
    static std::streamsize getSerializedSize(const RoadBoundaryList6902& list);
    static std::streamsize getSerializedSize(const RoadBoundaryIn6902& boundary);
    static std::streamsize getSerializedSize(const RoadBoundarySegmentIn6902& segments);
};

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
