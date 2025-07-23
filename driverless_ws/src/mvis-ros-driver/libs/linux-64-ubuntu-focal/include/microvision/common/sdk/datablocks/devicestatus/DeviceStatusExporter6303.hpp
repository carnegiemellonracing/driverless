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
//! \date Jan 30, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/devicestatus/DeviceStatus.hpp>
#include <microvision/common/sdk/datablocks/ExporterBase.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<DeviceStatus, DataTypeId::DataType_DeviceStatus6303>
  : public TypedExporter<DeviceStatus, DataTypeId::DataType_DeviceStatus6303>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;

private:
    static bool serialize(std::ostream& os, const SerialNumber& sn);
    static bool serialize(std::ostream& os, const Version448& version);
}; // DeviceStatusExporter6303

//==============================================================================

using DeviceStatusExporter6303 = Exporter<DeviceStatus, DataTypeId::DataType_DeviceStatus6303>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
