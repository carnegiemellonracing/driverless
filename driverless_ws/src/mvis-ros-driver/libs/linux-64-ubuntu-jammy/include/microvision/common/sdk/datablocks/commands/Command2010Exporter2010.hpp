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
//! \date Feb 28, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/datablocks/commands/Command2010Exporter2010.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

#include <microvision/common/sdk/misc/defines/defines.hpp>
#include <microvision/common/sdk/DataTypeId.hpp>
#include <microvision/common/sdk/datablocks/commands/Command2010.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

// Exporter for special data container 2010
template<>
class Exporter<Command2010, DataTypeId::DataType_Command2010>
  : public TypedExporter<Command2010, DataTypeId::DataType_Command2010>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase&) const override;

public:
    //========================================
    //!\brief Serialize command data to stream
    //!\param[in, out] os   Output data stream
    //!\param[out] c		Input container.
    //!\return \c True if serialization succeed, else: \c false
    //!\note This method is to be called from outside for serialization.
    //----------------------------------------
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // Command2010Exporter2010

//==============================================================================

using Command2010Exporter2010 = Exporter<Command2010, DataTypeId::DataType_Command2010>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
