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
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageNote6420.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessage64x0Exporter64x0Base.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::LogMessageNote6420, DataTypeId::DataType_LogNote6420>
  : public TypedExporter<microvision::common::sdk::LogMessageNote6420, DataTypeId::DataType_LogNote6420>,
    protected LogMessage64x0Exporter64x0Base<LogMessageNote6420>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // LogMessageNote6420Exporter6420

//==============================================================================

using LogMessageNote6420Exporter6420 = Exporter<LogMessageNote6420, DataTypeId::DataType_LogNote6420>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
