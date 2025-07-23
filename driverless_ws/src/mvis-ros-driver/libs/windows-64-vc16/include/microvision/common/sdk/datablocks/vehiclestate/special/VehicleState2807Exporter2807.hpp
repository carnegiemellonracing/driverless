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
#include <microvision/common/sdk/datablocks/vehiclestate/special/VehicleState2807.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<VehicleState2807, DataTypeId::DataType_VehicleStateBasic2807>
  : public TypedExporter<VehicleState2807, DataTypeId::DataType_VehicleStateBasic2807>
{
public:
    constexpr static std::streamsize serializedSize{90};

public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // VehicleState2807Exporter2807

//==============================================================================

using VehicleState2807Exporter2807 = Exporter<VehicleState2807, DataTypeId::DataType_VehicleStateBasic2807>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
