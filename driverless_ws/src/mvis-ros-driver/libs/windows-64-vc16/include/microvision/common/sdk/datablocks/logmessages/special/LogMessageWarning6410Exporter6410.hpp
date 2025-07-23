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
//! \date Mar 14, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageWarning6410.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessage64x0Exporter64x0Base.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::LogMessageWarning6410, DataTypeId::DataType_LogWarning6410>
  : public TypedExporter<microvision::common::sdk::LogMessageWarning6410, DataTypeId::DataType_LogWarning6410>,
    protected LogMessage64x0Exporter64x0Base<LogMessageWarning6410>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // LogMessageWarning6410Exporter6410

//==============================================================================

using LogMessageWarning6410Exporter6410 = Exporter<LogMessageWarning6410, DataTypeId::DataType_LogWarning6410>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
