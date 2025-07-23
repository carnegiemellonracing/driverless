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
//! \date Sep 02, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImu9004.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<GpsImu9004, DataTypeId::DataType_GpsImu9004>
  : public TypedExporter<GpsImu9004, DataTypeId::DataType_GpsImu9004>
{
public:
    static constexpr std::streamsize serializedSize{485};

public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // GpsImu9004Exporter9004

//==============================================================================

using GpsImu9004Exporter9004 = Exporter<GpsImu9004, DataTypeId::DataType_GpsImu9004>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
