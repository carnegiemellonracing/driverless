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
//! \date Jan 23, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/logpolygonlist2d/LogPolygonList2d.hpp>
#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<LogPolygonList2d, DataTypeId::DataType_LogPolygonList2dFloat6817>
  : public TypedExporter<LogPolygonList2d, DataTypeId::DataType_LogPolygonList2dFloat6817>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // LogPolygonList2dExporter6817

//==============================================================================

using LogPolygonList2dExporter6817 = Exporter<LogPolygonList2d, DataTypeId::DataType_LogPolygonList2dFloat6817>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
