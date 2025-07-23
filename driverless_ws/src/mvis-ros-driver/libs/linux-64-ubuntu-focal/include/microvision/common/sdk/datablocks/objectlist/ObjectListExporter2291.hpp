//==============================================================================
//! \file
//!
//! \brief Exports object type 0x2291 to general object container
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Apr 25, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectList.hpp>
#include <microvision/common/sdk/datablocks/objectlist/ObjectListExporter2281_2291.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<ObjectList, DataTypeId::DataType_ObjectList2291>
  : public TypedExporter<ObjectList, DataTypeId::DataType_ObjectList2291>, protected ObjectListExporter2281_2291
{
public:
    virtual ~Exporter() = default;

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override
    {
        return ObjectListExporter2281_2291::getSerializedSize(c);
    }

    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override
    {
        return ObjectListExporter2281_2291::serialize(os, c);
    }
}; // ObjectListExporter2291

//==============================================================================

using ObjectListExporter2291 = Exporter<ObjectList, DataTypeId::DataType_ObjectList2291>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
