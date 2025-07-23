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
//! \date May 22, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/logmessages/LogMessage.hpp>
#include <microvision/common/sdk/datablocks/logmessages/LogMessageExporter.hpp>
//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<LogMessage, DataTypeId::DataType_LogNote6420>
  : public TypedExporter<LogMessage, DataTypeId::DataType_LogNote6420>, protected LogMessageExporter<LogMessage>
{
public:
    //========================================
    //!\brief get size in bytes of serialized data
    //!\param[in]      c       Data container.
    //!\return \c True if serialization succeed, else: \c false
    //!\note This method is to be called from outside for serialization.
    //----------------------------------------
    virtual std::streamsize getSerializedSize(const DataContainerBase&) const override;

    //!\brief convert to byte stream (serialization)
    //!\param[in, out] os      Output data stream
    //!\param[in]      c       Data container.
    //!\return \c True if serialization succeeded, else: \c false
    //!\note This method is to be called from outside for serialization.
    virtual bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // LogMessageExporter6420

//==============================================================================

using LogMessageExporter6420 = Exporter<LogMessage, DataTypeId::DataType_LogNote6420>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
