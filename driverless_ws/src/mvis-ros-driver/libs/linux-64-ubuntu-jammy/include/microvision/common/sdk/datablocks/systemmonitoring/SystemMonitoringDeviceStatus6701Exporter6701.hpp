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
#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringDeviceStatus6701.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::SystemMonitoringDeviceStatus6701,
               DataTypeId::DataType_SystemMonitoringDeviceStatus6701>
  : public TypedExporter<microvision::common::sdk::SystemMonitoringDeviceStatus6701,
                         DataTypeId::DataType_SystemMonitoringDeviceStatus6701>
{
public:
    constexpr static const std::streamsize serializedSize{std::streamsize(
        sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint64_t))};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // SystemMonitoringDeviceStatus6701Exporter6701

//==============================================================================

using SystemMonitoringDeviceStatus6701Exporter6701
    = Exporter<microvision::common::sdk::SystemMonitoringDeviceStatus6701,
               DataTypeId::DataType_SystemMonitoringDeviceStatus6701>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
