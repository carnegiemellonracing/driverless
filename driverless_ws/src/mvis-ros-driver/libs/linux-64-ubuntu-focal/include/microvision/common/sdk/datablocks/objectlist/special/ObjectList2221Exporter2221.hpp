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
//! \date Jan 17, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2221.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<ObjectList2221, DataTypeId::DataType_ObjectList2221>
  : public TypedExporter<ObjectList2221, DataTypeId::DataType_ObjectList2221>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& container) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // ObjectList2221Exporter2221

//==============================================================================

using ObjectList2221Exporter2221 = Exporter<ObjectList2221, DataTypeId::DataType_ObjectList2221>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
