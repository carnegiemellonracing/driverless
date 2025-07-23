///==============================================================================
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
//! \date Jan 13, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/measurementlist/special/MeasurementList2821.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::MeasurementList2821, DataTypeId::DataType_MeasurementList2821>
  : public TypedExporter<microvision::common::sdk::MeasurementList2821, DataTypeId::DataType_MeasurementList2821>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase&) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // MeasurementList2821Exporter2821

//==============================================================================

using MeasurementList2821Exporter2821
    = Exporter<microvision::common::sdk::MeasurementList2821, DataTypeId::DataType_MeasurementList2821>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
