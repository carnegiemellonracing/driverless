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
//! \date Sep 03, 2021
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/gpsimu/GpsImu.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<GpsImu, DataTypeId::DataType_GpsImu9004> : public TypedExporter<GpsImu, DataTypeId::DataType_GpsImu9004>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // GpsImuExporter9004

//==============================================================================

using GpsImuExporter9004 = Exporter<GpsImu, DataTypeId::DataType_GpsImu9004>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
