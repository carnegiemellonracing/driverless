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
//! \date Jan 01, 2016
//! \brief Transformation helper class
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/TransformationMatrix2d.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectIn2281.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2807.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2808.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

class Transform final
{
public:
    static TransformationMatrix2d<float> getTransformationSystem(const ObjectIn2281& object);
    static TransformationMatrix2d<float> getTransformationSystem(const VehicleState2807& vs);
    static TransformationMatrix2d<float> getTransformationSystem(const VehicleState2808& vs);

    static TransformationMatrix2d<float> transformToGlobal(const TransformationMatrix2d<float>& ref2Global,
                                                           const TransformationMatrix2d<float>& rel2ref);
}; // Transform

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
