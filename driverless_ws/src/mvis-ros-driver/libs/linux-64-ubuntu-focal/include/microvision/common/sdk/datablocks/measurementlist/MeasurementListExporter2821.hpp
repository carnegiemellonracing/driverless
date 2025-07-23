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
//! \date Mar 25th, 2019
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/measurementlist/MeasurementList.hpp>
#include <microvision/common/sdk/datablocks/measurementlist/special/MeasurementList2821.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<MeasurementList, DataTypeId::DataType_MeasurementList2821>
  : public TypedExporter<MeasurementList, DataTypeId::DataType_MeasurementList2821>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // MeasurementListExporter2821

//==============================================================================

using MeasurementListExporter2821 = Exporter<MeasurementList, DataTypeId::DataType_MeasurementList2821>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
