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
//! \date Jan 31, 2019
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/ogpsimumessage/OGpsImuMessage.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<OGpsImuMessage, DataTypeId::DataType_OGpsImuMessage2610>
  : public TypedExporter<OGpsImuMessage, DataTypeId::DataType_OGpsImuMessage2610>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // OGpsImuMessageExporter2610

//==============================================================================

using OGpsImuMessageExporter2610
    = Exporter<microvision::common::sdk::OGpsImuMessage, DataTypeId::DataType_OGpsImuMessage2610>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
