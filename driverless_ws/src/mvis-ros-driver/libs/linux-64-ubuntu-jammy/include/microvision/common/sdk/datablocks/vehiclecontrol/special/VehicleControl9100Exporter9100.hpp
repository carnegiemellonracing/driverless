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
//! \date Mar 26, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include "VehicleControl9100.hpp"

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<VehicleControl9100, DataTypeId::DataType_VehicleControl9100>
  : public TypedExporter<VehicleControl9100, DataTypeId::DataType_VehicleControl9100>
{
public:
    static constexpr std::streamsize serializedBaseSize{32};

public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // VehicleControl9100Exporter9100

//==============================================================================

using VehicleControl9100Exporter9100 = Exporter<VehicleControl9100, DataTypeId::DataType_VehicleControl9100>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
