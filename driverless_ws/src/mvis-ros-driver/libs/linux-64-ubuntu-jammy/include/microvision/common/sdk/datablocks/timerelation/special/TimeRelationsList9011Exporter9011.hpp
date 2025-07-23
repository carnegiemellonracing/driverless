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
//! \date March 15, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationsList9011.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<TimeRelationsList9011, DataTypeId::DataType_TimeRelationsList9011>
  : public TypedExporter<TimeRelationsList9011, DataTypeId::DataType_TimeRelationsList9011>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // TimeRelationsList9011Exporter9011

//==============================================================================

using TimeRelationsList9011Exporter9011 = Exporter<TimeRelationsList9011, DataTypeId::DataType_TimeRelationsList9011>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
