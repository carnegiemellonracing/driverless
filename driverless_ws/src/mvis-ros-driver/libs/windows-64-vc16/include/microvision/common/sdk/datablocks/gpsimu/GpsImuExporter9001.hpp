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
//! \date Jan 31, 2019
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
class Exporter<GpsImu, DataTypeId::DataType_GpsImu9001> : public TypedExporter<GpsImu, DataTypeId::DataType_GpsImu9001>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // GpsImuExporter9001

//==============================================================================

using GpsImuExporter9001 = Exporter<GpsImu, DataTypeId::DataType_GpsImu9001>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
