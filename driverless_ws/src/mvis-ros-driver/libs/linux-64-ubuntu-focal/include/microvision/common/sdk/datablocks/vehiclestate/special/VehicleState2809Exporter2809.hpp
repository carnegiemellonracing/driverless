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
//! \date Jun 25, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2809.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<VehicleState2809, DataTypeId::DataType_VehicleStateBasic2809>
  : public TypedExporter<VehicleState2809, DataTypeId::DataType_VehicleStateBasic2809>
{
public:
    constexpr static const std::streamsize serializedSize{1150};

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
}; // VehicleState2809Exporter2809

//==============================================================================

using VehicleState2809Exporter2809 = Exporter<VehicleState2809, DataTypeId::DataType_VehicleStateBasic2809>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
