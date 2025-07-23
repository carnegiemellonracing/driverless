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
//! \date Sep 1, 2017
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/VehicleState.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::VehicleState, DataTypeId::DataType_VehicleStateBasic2808>
  : public TypedExporter<microvision::common::sdk::VehicleState, DataTypeId::DataType_VehicleStateBasic2808>
{
public:
    static constexpr const std::streamsize serializedSize{204};

public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& outStream, const DataContainerBase& c) const override;
}; // VehicleStateExporter2808

//==============================================================================

using VehicleStateExporter2808 = Exporter<VehicleState, DataTypeId::DataType_VehicleStateBasic2808>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
