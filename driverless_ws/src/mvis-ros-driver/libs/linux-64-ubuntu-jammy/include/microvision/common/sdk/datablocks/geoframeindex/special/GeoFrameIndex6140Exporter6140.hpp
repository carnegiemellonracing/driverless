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
#include <microvision/common/sdk/datablocks/geoframeindex/special/GeoFrameIndex6140.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<GeoFrameIndex6140, DataTypeId::DataType_GeoFrameIndex6140>
  : public TypedExporter<GeoFrameIndex6140, DataTypeId::DataType_GeoFrameIndex6140>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // GeoFrameIndex6140Exporter6140

//==============================================================================

using GeoFrameIndex6140Exporter6140 = Exporter<GeoFrameIndex6140, DataTypeId::DataType_GeoFrameIndex6140>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
