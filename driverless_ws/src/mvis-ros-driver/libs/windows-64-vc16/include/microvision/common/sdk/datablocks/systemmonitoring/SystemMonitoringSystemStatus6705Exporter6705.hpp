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
//! \date Jan 16, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringSystemStatus6705.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::SystemMonitoringSystemStatus6705,
               DataTypeId::DataType_SystemMonitoringSystemStatus6705>
  : public TypedExporter<microvision::common::sdk::SystemMonitoringSystemStatus6705,
                         DataTypeId::DataType_SystemMonitoringSystemStatus6705>
{
public:
    static constexpr const std::streamsize serializedSize{
        std::streamsize(sizeof(uint64_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t)
                        + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t))};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // SystemMonitoringSystemStatus6705Exporter6705

//==============================================================================

using SystemMonitoringSystemStatus6705Exporter6705
    = Exporter<microvision::common::sdk::SystemMonitoringSystemStatus6705,
               DataTypeId::DataType_SystemMonitoringSystemStatus6705>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
