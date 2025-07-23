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
//! \date Mar 1, 2018
//------------------------------------------------------------------------------

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/SpecialExporterBase.hpp>
#include <microvision/common/sdk/datablocks/commands/luxcommands/CommandLuxSetNtpTimestampSyncC.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class SpecialExporter<CommandLuxSetNtpTimestampSyncC> : public SpecialExporterBase<CommandCBase>
{
public:
    static constexpr int commandSize{14};

public:
    SpecialExporter() : SpecialExporterBase<CommandCBase>() {}

public:
    //========================================
    //!\brief Get the DataType of exporter/importer.
    //!\return The DataTypeId of the data this exporter/importer
    //!        can handle.
    //----------------------------------------
    virtual CommandCBase::KeyType getSpecialType() const override { return CommandLuxSetNtpTimestampSyncC::key; }

public:
    //========================================
    //!\brief Get serializable size of data from exporter/importer.
    //!\return Number of Bytes used by data type.
    //----------------------------------------
    virtual std::streamsize getSerializedSize(const CommandCBase& c) const override;

    virtual bool serialize(std::ostream& outStream, const CommandCBase& c) const override;
}; // CommandLuxSetNtpTimestampSync2010Exporter2010

//==============================================================================

using CommandLuxSetNtpTimestampSync2010Exporter2010 = SpecialExporter<CommandLuxSetNtpTimestampSyncC>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
