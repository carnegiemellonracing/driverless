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
#include <microvision/common/sdk/datablocks/objectlist/special/ObjectList2270.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<ObjectList2270, DataTypeId::DataType_ObjectList2270>
  : public TypedExporter<ObjectList2270, DataTypeId::DataType_ObjectList2270>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // ObjectList2270Exporter2270

//==============================================================================

using ObjectList2270Exporter2270 = Exporter<ObjectList2270, DataTypeId::DataType_ObjectList2270>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
