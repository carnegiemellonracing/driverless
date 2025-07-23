//==============================================================================
//! \file
//!
//! \brief Serialize PointCloud7500 data container into stream.
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
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7500.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

template<>
class Exporter<PointCloud7500, DataTypeId::DataType_PointCloud7500>
  : public TypedExporter<PointCloud7500, DataTypeId::DataType_PointCloud7500>
{
public:
    virtual std::streamsize getSerializedSize(const DataContainerBase& c) const override;

public:
    virtual bool serialize(std::ostream& os, const DataContainerBase& c) const override;
}; // PointCloud7500Exporter7500

//==============================================================================

using PointCloud7500Exporter7500 = Exporter<PointCloud7500, DataTypeId::DataType_PointCloud7500>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
