//==============================================================================
//! \file
//!
//! \brief Serialize LdmiRawFrame2354 data container into stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jul 6th, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2354.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Serialize LdmiRawFrame2354 data container into stream.
//------------------------------------------------------------------------------
template<>
class Exporter<LdmiRawFrame2354, DataTypeId::DataType_LdmiRawFrame2354>
  : public TypedExporter<LdmiRawFrame2354, DataTypeId::DataType_LdmiRawFrame2354>
{
public:
    //========================================
    //!\brief Get size in bytes of serialized data.
    //!\param[in] c  Data container.
    //!\return Size in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, else: \c false
    //!\note This method is to be called from outside for serialization.
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // LdmiRawFrame2354Exporter2354

//==============================================================================

using LdmiRawFrame2354Exporter2354 = Exporter<LdmiRawFrame2354, DataTypeId::DataType_LdmiRawFrame2354>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
