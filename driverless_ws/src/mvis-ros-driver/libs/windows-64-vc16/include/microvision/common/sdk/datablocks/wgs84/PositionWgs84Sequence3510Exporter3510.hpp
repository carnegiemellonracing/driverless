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
//! \date Sept 03, 2018
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/wgs84/PositionWgs84Sequence3510.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<PositionWgs84Sequence3510, DataTypeId::DataType_PositionWgs84Sequence3510>
  : public TypedExporter<PositionWgs84Sequence3510, DataTypeId::DataType_PositionWgs84Sequence3510>
{
public:
    std::streamsize getSerializedSize(const DataContainerBase& c) const override;
    bool serialize(std::ostream& os, const DataContainerBase& c) const override;

}; //PositionWgs84Sequence3510Exporter3510

//==============================================================================

using PositionWgs84Sequence3510Exporter3510
    = Exporter<PositionWgs84Sequence3510, DataTypeId::DataType_PositionWgs84Sequence3510>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
