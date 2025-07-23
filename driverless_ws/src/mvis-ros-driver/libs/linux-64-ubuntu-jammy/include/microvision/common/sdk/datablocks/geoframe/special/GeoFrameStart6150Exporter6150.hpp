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
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/geoframe/special/GeoFrameStart6150.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<GeoFrameStart6150, DataTypeId::DataType_GeoFrameStart6150>
  : public TypedExporter<GeoFrameStart6150, DataTypeId::DataType_GeoFrameStart6150>
{
public:
    static constexpr std::streamsize serializedSize{32};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // GeoFrameStart6150Exporter6150

//==============================================================================

using GeoFrameStart6150Exporter6150 = Exporter<GeoFrameStart6150, DataTypeId::DataType_GeoFrameStart6150>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
