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
//! \date March 14, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/timerelation/special/TimeRelationsList9010.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::TimeRelationsList9010, DataTypeId::DataType_TimeRelationsList9010>
  : public TypedExporter<microvision::common::sdk::TimeRelationsList9010, DataTypeId::DataType_TimeRelationsList9010>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // TimeRelationsList9010Exporter9010

//==============================================================================

using TimeRelationsList9010Exporter9010
    = Exporter<microvision::common::sdk::TimeRelationsList9010, DataTypeId::DataType_TimeRelationsList9010>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
