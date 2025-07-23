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
//! \date Jan 15, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/wgs84/PositionWgs84_2604.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<microvision::common::sdk::PositionWgs84_2604, DataTypeId::DataType_PositionWgs84_2604>
  : public TypedExporter<microvision::common::sdk::PositionWgs84_2604, DataTypeId::DataType_PositionWgs84_2604>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // PositionWgs84_2604Exporter2604

//==============================================================================

using PositionWgs84_2604Exporter2604
    = Exporter<microvision::common::sdk::PositionWgs84_2604, DataTypeId::DataType_PositionWgs84_2604>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
