//==============================================================================
//! \file
//!
//! \brief Serialize PointCloud7511 data container into stream as PointCloud7510.
//!
//! $$MICROVISION_LICENSE_BEGIN$$
//! Copyright (c) 2025 MicroVision, Inc., Redmond, U.S.A.
//! All Rights Reserved.
//!
//! For more details, please refer to the accompanying file
//! License.txt.
//! $$MICROVISION_LICENSE_END$$
//!
//! \date Nov 08, 2022
//------------------------------------------------------------------------------

#pragma once

//==============================================================================

#include <microvision/common/sdk/misc/defines/defines.hpp>

#include <microvision/common/sdk/datablocks/ExporterBase.hpp>
#include <microvision/common/sdk/datablocks/pointcloud/special/PointCloud7511.hpp>

//==============================================================================
namespace microvision {
namespace common {
namespace sdk {
//==============================================================================

//==============================================================================
//! \brief This class is used to export a point cloud tile and to serialize it into a
//! binary idc data block.
//------------------------------------------------------------------------------
template<>
class Exporter<PointCloud7511, DataTypeId::DataType_PointCloud7510>
  : public TypedExporter<PointCloud7511, DataTypeId::DataType_PointCloud7510>
{
public:
    //========================================
    //!\brief Get size in bytes of serialized data.
    //!\param[in] c  Data container.
    //!\return Size in bytes.
    //----------------------------------------
    std::streamsize getSerializedSize(const DataContainerBase&) const override;

    //========================================
    //!\brief Convert to byte stream (serialization).
    //!\param[in, out] os  Output data stream.
    //!\param[in]      c   Data container.
    //!\return \c True if serialization succeeded, otherwise: \c false
    //!\note This method is to be called from outside for serialization.
    //----------------------------------------
    bool serialize(std::ostream& os, const DataContainerBase& importContainer) const override;
}; // PointCloud7511Exporter7510

//==============================================================================

using PointCloud7511Exporter7510 = Exporter<PointCloud7511, DataTypeId::DataType_PointCloud7510>;

//==============================================================================
} // namespace sdk
} // namespace common
} // namespace microvision
//==============================================================================
