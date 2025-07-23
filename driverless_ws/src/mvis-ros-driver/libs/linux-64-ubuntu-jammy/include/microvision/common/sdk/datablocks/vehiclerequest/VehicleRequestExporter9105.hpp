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
//! \date Nov 6, 2020
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/vehiclerequest/VehicleRequest.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Serialize VehicleRequest data container into 9105 stream.
//------------------------------------------------------------------------------
template<>
class Exporter<VehicleRequest, DataTypeId::DataType_VehicleRequest9105>
  : public TypedExporter<VehicleRequest, DataTypeId::DataType_VehicleRequest9105>
{
public:
    //========================================
    //!\brief get size in bytes of serialized data
    //!\param[in] c  Data container.
    //!\return Size in bytes
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

    //========================================
    //!\brief convert to byte stream (serialization)
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, else: \c false
    //!\note This method is to be called from outside for serialization.
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // VehicleRequestExporter9105

//==============================================================================

using VehicleRequestExporter9105 = Exporter<VehicleRequest, DataTypeId::DataType_VehicleRequest9105>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
