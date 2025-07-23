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
//! \date Mar 20, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/objectassociationlist/special/ObjectAssociationList4001.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::ObjectAssociationList4001, DataTypeId::DataType_ObjectAssociationList4001>
  : public TypedExporter<microvision::common::sdk::ObjectAssociationList4001,
                         DataTypeId::DataType_ObjectAssociationList4001>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase&) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // ObjectAssociationList4001Exporter4001

//==============================================================================

using ObjectAssociationList4001Exporter4001
    = Exporter<ObjectAssociationList4001, DataTypeId::DataType_ObjectAssociationList4001>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
