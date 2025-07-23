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
//! \date Nov 20, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/lanemarking/special/LaneMarkingIn6901.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
namespace lanes {
//==============================================================================

//==============================================================================
//! \brief All segments representing one lane marking line.
//!
//! \sa microvision::common::sdk::LaneMarkingList
//------------------------------------------------------------------------------
using LaneMarking = LaneMarkingIn6901;

//==============================================================================
} // namespace lanes
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
