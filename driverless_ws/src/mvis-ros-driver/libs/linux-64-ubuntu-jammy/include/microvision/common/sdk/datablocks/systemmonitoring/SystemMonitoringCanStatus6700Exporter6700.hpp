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
#include <microvision/common/sdk/datablocks/systemmonitoring/SystemMonitoringCanStatus6700.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<SystemMonitoringCanStatus6700, DataTypeId::DataType_SystemMonitoringCanStatus6700>
  : public TypedExporter<SystemMonitoringCanStatus6700, DataTypeId::DataType_SystemMonitoringCanStatus6700>
{
public:
    constexpr static const std::streamsize serializedSize{static_cast<std::streamsize>(
        sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint64_t))};

public:
    virtual std::streamsize getSerializedSize(const DataContainerBase&) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // SystemMonitoringCanStatus6700Exporter6700

//==============================================================================

using SystemMonitoringCanStatus6700Exporter6700
    = Exporter<SystemMonitoringCanStatus6700, DataTypeId::DataType_SystemMonitoringCanStatus6700>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
