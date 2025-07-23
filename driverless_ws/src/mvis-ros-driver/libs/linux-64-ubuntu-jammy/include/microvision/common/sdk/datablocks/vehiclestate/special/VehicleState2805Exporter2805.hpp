
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

//==============================================================================

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2805.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<VehicleState2805, DataTypeId::DataType_VehicleStateBasic2805>
  : public TypedExporter<VehicleState2805, DataTypeId::DataType_VehicleStateBasic2805>
{
public:
    constexpr static const std::streamsize serializedSize{46};

public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // VehicleState2805Exporter2805

//==============================================================================

using VehicleState2805Exporter2805
    = Exporter<microvision::common::sdk::VehicleState2805, DataTypeId::DataType_VehicleStateBasic2805>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
