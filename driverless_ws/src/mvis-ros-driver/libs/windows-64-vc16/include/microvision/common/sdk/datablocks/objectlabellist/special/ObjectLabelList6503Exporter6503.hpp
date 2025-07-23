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
//! \date Jan 15, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/objectlabellist/special/ObjectLabelList6503.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::ObjectLabelList6503, DataTypeId::DataType_ObjectLabel6503>
  : public TypedExporter<microvision::common::sdk::ObjectLabelList6503, DataTypeId::DataType_ObjectLabel6503>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase&) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // ObjectLabelList6503Exporter6503

//==============================================================================

using ObjectLabelList6503Exporter6503
    = Exporter<microvision::common::sdk::ObjectLabelList6503, DataTypeId::DataType_ObjectLabel6503>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
