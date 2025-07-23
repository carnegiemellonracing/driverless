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
//! \date Nov 19, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/Point3dWithVariance.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//! \brief One measurement point in the lane markings.
//!
//! \sa microvision::common::sdk::LaneMarkingList6901
//------------------------------------------------------------------------------
using LaneMarkingPointIn6901 = Point3dWithVariance<float>;

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
