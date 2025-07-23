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
//! \date Sept 05, 2018
//------------------------------------------------------------------------------
#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/missionresponse/special/MissionResponse3540.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<MissionResponse3540, DataTypeId::DataType_MissionResponse3540>
  : public TypedExporter<MissionResponse3540, DataTypeId::DataType_MissionResponse3540>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;

}; //MissionResponse3540Exporter3540

//==============================================================================

using MissionResponse3540Exporter3540 = Exporter<MissionResponse3540, DataTypeId::DataType_MissionResponse3540>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
