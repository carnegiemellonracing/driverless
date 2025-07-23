//==============================================================================
//! \file
//!
//! \brief Exporter for MavinRawFrame2360 in binary format for MavinRawFrame2360.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Mar 1th, 2023
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/mavinRaw/special/MavinRawFrame2360.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Serialize MavinRawFrame2360 data container into stream.
//------------------------------------------------------------------------------
template<>
class Exporter<MavinRawFrame2360, DataTypeId::DataType_MavinRawFrame2360>
  : public TypedExporter<MavinRawFrame2360, DataTypeId::DataType_MavinRawFrame2360>
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
}; // MavinRawFrame2360Exporter2360

//==============================================================================

using MavinRawFrame2360Exporter2360 = Exporter<MavinRawFrame2360, DataTypeId::DataType_MavinRawFrame2360>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
