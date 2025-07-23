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
//! \date Apr 29, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/timerecord/TimeRecord.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<TimeRecord, DataTypeId::DataType_TimeRecord9000>
  : public TypedExporter<TimeRecord, DataTypeId::DataType_TimeRecord9000>
{
public:
    Exporter() = default;

    Exporter(const Exporter&) = delete;
    Exporter& operator=(const Exporter&) = delete;

public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // TimeRecordExporter9000

//==============================================================================

using TimeRecordExporter9000 = Exporter<TimeRecord, DataTypeId::DataType_TimeRecord9000>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
