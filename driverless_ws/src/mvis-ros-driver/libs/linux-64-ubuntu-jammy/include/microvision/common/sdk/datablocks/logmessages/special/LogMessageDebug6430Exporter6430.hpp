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
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageDebug6430.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessage64x0Exporter64x0Base.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<LogMessageDebug6430, DataTypeId::DataType_LogDebug6430>
  : public TypedExporter<LogMessageDebug6430, DataTypeId::DataType_LogDebug6430>,
    protected LogMessage64x0Exporter64x0Base<LogMessageDebug6430>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // LogMessageDebug6430Exporter6430

//==============================================================================

using LogMessageDebug6430Exporter6430 = Exporter<LogMessageDebug6430, DataTypeId::DataType_LogDebug6430>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
