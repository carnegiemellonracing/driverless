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
//! \date Jan 19, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2808.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<VehicleState2808, DataTypeId::DataType_VehicleStateBasic2808>
  : public TypedExporter<VehicleState2808, DataTypeId::DataType_VehicleStateBasic2808>
{
public:
    constexpr static const std::streamsize serializedSize{204};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // VehicleState2808Exporter2808

//==============================================================================

using VehicleState2808Exporter2808 = Exporter<VehicleState2808, DataTypeId::DataType_VehicleStateBasic2808>;

//==============================================================================

template<>
void writeBE<VehicleState2808::VehicleStateSource>(std::ostream& os, const VehicleState2808::VehicleStateSource& vss);

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
