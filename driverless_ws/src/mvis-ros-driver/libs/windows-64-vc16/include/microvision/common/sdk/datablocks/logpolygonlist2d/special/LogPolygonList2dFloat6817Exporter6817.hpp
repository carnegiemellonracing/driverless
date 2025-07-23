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
//! \date Mar 26, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/logpolygonlist2d/special/LogPolygonList2dFloat6817.hpp>
#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<LogPolygonList2dFloat6817, DataTypeId::DataType_LogPolygonList2dFloat6817>
  : public TypedExporter<LogPolygonList2dFloat6817, DataTypeId::DataType_LogPolygonList2dFloat6817>
{
public:
    static constexpr std::streamsize serializedBaseSize{2};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // LogPolygonList2dFloat6817Exporter6817

//==============================================================================

using LogPolygonList2dFloat6817Exporter6817
    = Exporter<LogPolygonList2dFloat6817, DataTypeId::DataType_LogPolygonList2dFloat6817>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
