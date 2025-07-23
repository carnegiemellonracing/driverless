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

#include <microvision/common/sdk/Point3dWithVariance.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief One measurement point in the road boundaries.
//!
//! \sa microvision::common::sdk::RoadBoundaryList6902
//------------------------------------------------------------------------------
using RoadBoundaryPointIn6902 = Point3dWithVariance<float>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
