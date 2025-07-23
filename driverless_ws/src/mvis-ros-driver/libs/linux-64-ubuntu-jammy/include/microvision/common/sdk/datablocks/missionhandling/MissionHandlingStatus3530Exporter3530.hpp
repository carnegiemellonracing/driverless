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
//! \date 02.November 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/missionhandling/MissionHandlingStatus3530.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<MissionHandlingStatus3530, DataTypeId::DataType_MissionHandlingStatus3530>
  : public TypedExporter<MissionHandlingStatus3530, DataTypeId::DataType_MissionHandlingStatus3530>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // MissionHandlingStatus3530

//==============================================================================

using MissionHandlingStatus3530Exporter3530
    = Exporter<microvision::common::sdk::MissionHandlingStatus3530, DataTypeId::DataType_MissionHandlingStatus3530>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
