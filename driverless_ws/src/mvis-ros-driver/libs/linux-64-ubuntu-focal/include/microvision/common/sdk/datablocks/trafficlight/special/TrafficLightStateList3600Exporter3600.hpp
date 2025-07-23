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
//! \date Aug 29, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/trafficlight/special/TrafficLightStateList3600.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<TrafficLightStateList3600, DataTypeId::DataType_TrafficLight3600>
  : public TypedExporter<TrafficLightStateList3600, DataTypeId::DataType_TrafficLight3600>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;

}; // TrafficLightStateList3600Exporter3600

//==============================================================================

using TrafficLightStateList3600Exporter3600
    = Exporter<TrafficLightStateList3600, DataTypeId::DataType_TrafficLight3600>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
