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
//! \date Feb 16, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/SpecialRegisteredImporter.hpp>
#include <microvision/common/sdk/datablocks/commands/ecucommands/CommandEcuAppBaseCtrlC.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class SpecialImporter<Command2010, DataTypeId::DataType_Command2010, CommandEcuAppBaseCtrlC>
  : public SpecialRegisteredImporter<Command2010, DataTypeId::DataType_Command2010, CommandEcuAppBaseCtrlC>
{
public:
    static constexpr uint8_t commandBaseSize{4};

    //========================================
    //! \brief Maximum size of the data string.
    //--------------------------------------
    static constexpr uint16_t maxDataStringSize{10000};

public:
    SpecialImporter()
      : SpecialRegisteredImporter<Command2010, DataTypeId::DataType_Command2010, CommandEcuAppBaseCtrlC>()
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
}; // SpecialCommand2010Importer2010

//==============================================================================

using CommandEcuAppBaseCtrl2010Importer2010
    = SpecialImporter<Command2010, DataTypeId::DataType_Command2010, CommandEcuAppBaseCtrlC>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
