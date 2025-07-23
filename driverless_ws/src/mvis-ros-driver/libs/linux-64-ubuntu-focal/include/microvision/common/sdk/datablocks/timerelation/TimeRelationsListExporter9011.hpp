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
//! \date Apr 5, 2018
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/timerelation/TimeRelationsList.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::TimeRelationsList, DataTypeId::DataType_TimeRelationsList9011>
  : public TypedExporter<microvision::common::sdk::TimeRelationsList, DataTypeId::DataType_TimeRelationsList9011>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& outStream, const DataContainerBase& c) const override;
}; // VehicleStateExporter2805

//==============================================================================

using TimeRelationsListExporter9011 = Exporter<TimeRelationsList, DataTypeId::DataType_TimeRelationsList9011>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
