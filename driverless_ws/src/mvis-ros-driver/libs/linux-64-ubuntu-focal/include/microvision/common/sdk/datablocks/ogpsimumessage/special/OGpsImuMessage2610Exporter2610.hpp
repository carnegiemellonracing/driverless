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
//! \date Jan 29, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/ogpsimumessage/special/OGpsImuMessage2610.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<OGpsImuMessage2610, DataTypeId::DataType_OGpsImuMessage2610>
  : public TypedExporter<OGpsImuMessage2610, DataTypeId::DataType_OGpsImuMessage2610>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // OGpsImuMessage2610Exporter2610

//==============================================================================

using OGpsImuMessage2610Exporter2610
    = Exporter<microvision::common::sdk::OGpsImuMessage2610, DataTypeId::DataType_OGpsImuMessage2610>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
