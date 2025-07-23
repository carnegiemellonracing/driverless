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

#include <microvision/common/sdk/datablocks/vehiclerequest/special/VehicleRequest9105.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief Serialize VehicleRequest9105 data container into stream.
//------------------------------------------------------------------------------
template<>
class Exporter<VehicleRequest9105, DataTypeId::DataType_VehicleRequest9105>
  : public TypedExporter<VehicleRequest9105, DataTypeId::DataType_VehicleRequest9105>
{
public:
    //========================================
    //! \brief Serialized size of this data container.
    //----------------------------------------
    static constexpr std::streamsize serializedBaseSize{2};

public:
    //========================================
    //!\brief Get size in bytes of serialized data.
    //!\param[in] c  Data container.
    //!\return Size in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, else: \c false
    //!\note This method is to be called from outside for serialization.
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // VehicleRequest9105Exporter9105

//==============================================================================

using VehicleRequest9105Exporter9105 = Exporter<VehicleRequest9105, DataTypeId::DataType_VehicleRequest9105>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
