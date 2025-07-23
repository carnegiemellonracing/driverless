//==============================================================================
//! \file
//!
//! \brief Serialize LdmiRawFrame2353 data container into stream.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Jun 28th, 2021
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/ldmiRaw/special/LdmiRawFrame2353.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Serialize LdmiRawFrame2353 data container into stream.
//------------------------------------------------------------------------------
template<>
class Exporter<LdmiRawFrame2353, DataTypeId::DataType_LdmiRawFrame2353>
  : public TypedExporter<LdmiRawFrame2353, DataTypeId::DataType_LdmiRawFrame2353>
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
}; // LdmiRawFrame2353Exporter2353

//==============================================================================

using LdmiRawFrame2353Exporter2353 = Exporter<LdmiRawFrame2353, DataTypeId::DataType_LdmiRawFrame2353>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
