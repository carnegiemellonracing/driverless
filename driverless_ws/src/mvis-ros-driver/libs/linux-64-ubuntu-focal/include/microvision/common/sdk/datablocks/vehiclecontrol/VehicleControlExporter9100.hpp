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
//! \date May 23, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/vehiclecontrol/VehicleControl.hpp>
#include <microvision/common/sdk/datablocks/vehiclecontrol/special/VehicleControl9100.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<VehicleControl, DataTypeId::DataType_VehicleControl9100>
  : public TypedExporter<VehicleControl, DataTypeId::DataType_VehicleControl9100>
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
}; // VehicleControlExporter9100

//==============================================================================

using VehicleControlExporter9100 = Exporter<VehicleControl, DataTypeId::DataType_VehicleControl9100>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
