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
//! \date Mar 23, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

#include <microvision/common/sdk/datablocks/gpsimu/special/GpsImu9001.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<GpsImu9001, DataTypeId::DataType_GpsImu9001>
  : public TypedExporter<GpsImu9001, DataTypeId::DataType_GpsImu9001>
{
public:
    static constexpr std::streamsize serializedSize{413};

public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // GpsImu9001Exporter9001

//==============================================================================

using GpsImu9001Exporter9001 = Exporter<GpsImu9001, DataTypeId::DataType_GpsImu9001>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
