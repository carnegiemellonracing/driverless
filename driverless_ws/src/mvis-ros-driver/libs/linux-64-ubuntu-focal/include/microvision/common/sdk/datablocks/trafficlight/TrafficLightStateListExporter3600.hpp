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
//! \date 09.November 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/trafficlight/TrafficLightStateList.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<TrafficLightStateList, DataTypeId::DataType_TrafficLight3600>
  : public TypedExporter<TrafficLightStateList, DataTypeId::DataType_TrafficLight3600>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;

}; // TrafficLightStateListExporter3600

//==============================================================================

using TrafficLightStateListExporter3600 = Exporter<TrafficLightStateList, DataTypeId::DataType_TrafficLight3600>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
