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
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessageError6400.hpp>
#include <microvision/common/sdk/datablocks/logmessages/special/LogMessage64x0Exporter64x0Base.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::LogMessageError6400, DataTypeId::DataType_LogError6400>
  : public TypedExporter<microvision::common::sdk::LogMessageError6400, DataTypeId::DataType_LogError6400>,
    protected LogMessage64x0Exporter64x0Base<LogMessageError6400>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // LogMessageError6400Exporter6400

//==============================================================================

using LogMessageError6400Exporter6400 = Exporter<LogMessageError6400, DataTypeId::DataType_LogError6400>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
