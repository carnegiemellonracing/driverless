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
//! \date Jun 24, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/special/DeviceStatus6320.hpp>
#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<DeviceStatus6320, DataTypeId::DataType_DeviceStatus6320>
  : public TypedExporter<DeviceStatus6320, DataTypeId::DataType_DeviceStatus6320>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;

public:
    static bool serialize(std::ostream& os, const ErrorIn6320& error);
}; // DeviceStatus6320Exporter6320

//==============================================================================

using DeviceStatus6320Exporter6320 = Exporter<DeviceStatus6320, DataTypeId::DataType_DeviceStatus6320>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
