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
//! \date Feb 19, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/SpecialRegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/commands/luxcommands/CommandLuxSetNtpTimestampSyncC.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class SpecialImporter<Command2010, DataTypeId::DataType_Command2010, CommandLuxSetNtpTimestampSyncC>
  : public SpecialRegisteredImporter<Command2010, DataTypeId::DataType_Command2010, CommandLuxSetNtpTimestampSyncC>
{
public:
    static constexpr int commandSize{14};

public:
    SpecialImporter()
      : SpecialRegisteredImporter<Command2010, DataTypeId::DataType_Command2010, CommandLuxSetNtpTimestampSyncC>()
    {}
    SpecialImporter(const SpecialImporter&) = delete;
    SpecialImporter& operator=(const SpecialImporter&) = delete;

public:
    virtual std::streamsize getSerializedSize(const CommandCBase& s) const override;

    //========================================
    //!\brief convert data from source to target type (deserialization)
    //!\param[in, out] is      Input data stream
    //!\param[out]     c       Output container.
    //!\param[in]      header  idc dataHeader
    //!\return \c True if serialization succeed, else: \c false
    //!\note This method is to be called from outside for deserialization.
    //----------------------------------------
    virtual bool deserialize(std::istream& is, CommandCBase& s, const IdcDataHeader& header) const override;
}; // CommandLuxSetNtpTimestampSync2010Importer2010

//==============================================================================

using CommandLuxSetNtpTimestampSync2010Importer2010
    = SpecialImporter<Command2010, DataTypeId::DataType_Command2010, CommandLuxSetNtpTimestampSyncC>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
